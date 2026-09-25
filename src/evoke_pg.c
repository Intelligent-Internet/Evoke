#include "postgres.h"

#include "fmgr.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "varatt.h"

#include "evoke_core.h"
#include "evoke_storage.h"

PG_MODULE_MAGIC;

static void
evoke_validate_serialized_bytes(
    const uint8_t *bytes,
    size_t len
)
{
    evoke_index index;
    evoke_status status;

    evoke_index_init(&index);
    status = evoke_deserialize_index(bytes, len, &index);
    if (status != EVOKE_OK)
    {
        ereport(ERROR, (errmsg("invalid evoke_index value: %s",
                               evoke_strerror(status))));
    }
    evoke_index_free(&index);
}

PG_FUNCTION_INFO_V1(evoke_in);
Datum
evoke_in(PG_FUNCTION_ARGS)
{
    bytea *result;

    result = DatumGetByteaPP(
        DirectFunctionCall1(byteain, CStringGetDatum(PG_GETARG_CSTRING(0)))
    );

    evoke_validate_serialized_bytes(
        (const uint8_t *) VARDATA_ANY(result),
        (size_t) VARSIZE_ANY_EXHDR(result)
    );
    PG_RETURN_BYTEA_P(result);
}

PG_FUNCTION_INFO_V1(evoke_out);
Datum
evoke_out(PG_FUNCTION_ARGS)
{
    bytea *raw = PG_GETARG_BYTEA_PP(0);

    evoke_validate_serialized_bytes(
        (const uint8_t *) VARDATA_ANY(raw),
        (size_t) VARSIZE_ANY_EXHDR(raw)
    );
    PG_RETURN_CSTRING(
        DatumGetCString(DirectFunctionCall1(byteaout, PG_GETARG_DATUM(0)))
    );
}

PG_FUNCTION_INFO_V1(evoke_recv);
Datum
evoke_recv(PG_FUNCTION_ARGS)
{
    bytea *result;

    result = DatumGetByteaPP(DirectFunctionCall1(bytearecv, PG_GETARG_DATUM(0)));

    evoke_validate_serialized_bytes(
        (const uint8_t *) VARDATA_ANY(result),
        (size_t) VARSIZE_ANY_EXHDR(result)
    );
    PG_RETURN_BYTEA_P(result);
}

PG_FUNCTION_INFO_V1(evoke_send);
Datum
evoke_send(PG_FUNCTION_ARGS)
{
    bytea *raw = PG_GETARG_BYTEA_PP(0);

    evoke_validate_serialized_bytes(
        (const uint8_t *) VARDATA_ANY(raw),
        (size_t) VARSIZE_ANY_EXHDR(raw)
    );
    PG_RETURN_BYTEA_P(DatumGetByteaPP(DirectFunctionCall1(
        byteasend,
        PG_GETARG_DATUM(0)
    )));
}
