#include "postgres.h"

#include <errno.h>
#include <stdlib.h>

#include "miscadmin.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#if PG_VERSION_NUM >= 180000
#include "utils/wait_classes.h"
#else
#include "utils/wait_event.h"
#endif

#include "evoke_am_test_support.h"

void
evoke_am_test_pause_ms(
    const char *setting_name,
    const char *description
)
{
    const char *setting = GetConfigOptionByName(
        setting_name,
        NULL,
        true
    );
    char *end = NULL;
    long pause_ms;
    long remaining_ms;
    TimestampTz started;

    if (setting == NULL || setting[0] == '\0')
    {
        return;
    }
    errno = 0;
    pause_ms = strtol(setting, &end, 10);
    if (errno != 0 || end == setting || *end != '\0' ||
        pause_ms < 0 || pause_ms > 10000)
    {
        ereport(
            ERROR,
            (errmsg(
                "%s must be between 0 and 10000",
                setting_name
            ))
        );
    }
    if (pause_ms == 0)
    {
        return;
    }
    if (!superuser_arg(GetOuterUserId()))
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
                errmsg("%s test pause is superuser-only", description)
            )
        );
    }
    started = GetCurrentTimestamp();
    remaining_ms = pause_ms;
    while (remaining_ms > 0)
    {
        TimestampTz now;
        int64 elapsed_ms;
        long elapsed_secs;
        int elapsed_usecs;

        ResetLatch(&MyProc->procLatch);
        (void) WaitLatch(
            &MyProc->procLatch,
            WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
            remaining_ms,
            PG_WAIT_EXTENSION
        );
        CHECK_FOR_INTERRUPTS();
        now = GetCurrentTimestamp();
        TimestampDifference(
            started,
            now,
            &elapsed_secs,
            &elapsed_usecs
        );
        elapsed_ms = ((int64) elapsed_secs * 1000) +
            (elapsed_usecs / 1000);
        if (elapsed_ms >= pause_ms)
        {
            break;
        }
        remaining_ms = pause_ms - (long) elapsed_ms;
    }
}

void
evoke_am_test_error_if_enabled(
    const char *setting_name,
    const char *description
)
{
    if (evoke_am_test_setting_enabled(setting_name, description))
    {
        ereport(ERROR, (errmsg("injected evoke %s error", description)));
    }
}

bool
evoke_am_test_setting_enabled(
    const char *setting_name,
    const char *description
)
{
    const char *setting = GetConfigOptionByName(
        setting_name,
        NULL,
        true
    );
    bool enabled = false;

    if (setting == NULL || setting[0] == '\0')
    {
        return false;
    }
    if (!parse_bool(setting, &enabled))
    {
        ereport(ERROR, (errmsg("%s must be boolean", setting_name)));
    }
    if (!enabled)
    {
        return false;
    }
    if (!superuser_arg(GetOuterUserId()))
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
                errmsg("%s failure injection is superuser-only", description)
            )
        );
    }
    return true;
}
