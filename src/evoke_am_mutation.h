#ifndef EVOKE_AM_MUTATION_H
#define EVOKE_AM_MUTATION_H

#include "postgres.h"

#include "access/transam.h"
#include "access/xlogdefs.h"
#include "storage/bufpage.h"
#include "utils/rel.h"

#include "evoke_am_meta.h"
#include "evoke_segment_pages.h"

#define EVOKE_AM_ABORTED_DELTA_XID BootstrapTransactionId

typedef enum evoke_am_delta_xid_state
{
    EVOKE_AM_DELTA_XID_COMMITTED = 0,
    EVOKE_AM_DELTA_XID_ABORTED,
    EVOKE_AM_DELTA_XID_UNRESOLVED
} evoke_am_delta_xid_state;

uint32 evoke_am_active_l0_rotation_record_limit(void);
bool evoke_am_active_l0_checkpoint_due(
    const evoke_segment_read_root *root
);
uint32 evoke_am_accelerator_refresh_record_limit(void);
bool evoke_am_delta_record_states(
    const TransactionId *record_xids,
    evoke_am_delta_xid_state *states_out,
    uint32 record_count
);
bool evoke_am_l0_expected_source_matches(
    uint8 record_kind,
    const evoke_document_cow_record *expected,
    const evoke_document_cow_record *current
);

void evoke_am_l0_require_meta_root(
    const evoke_am_meta_page *meta,
    const evoke_am_meta_page *expected
);
void evoke_am_l0_store_root(
    Page meta_page,
    const evoke_segment_read_root *root
);
bool evoke_am_mutation_publish_cow_manifest(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *next_manifest,
    const evoke_segment_cow_result *cow_result,
    bool seal_pending,
    bool require_exact_frontiers,
    bool publish_fsm_handoff
);
XLogRecPtr evoke_am_l0_rotate_active_locked(
    Relation index_relation,
    evoke_am_meta_page *expected_meta,
    evoke_segment_read_root *root
);
bool evoke_am_mutation_append_l0_record(
    Relation index_relation,
    evoke_l0_record *record,
    const evoke_document_cow_record *expected_source,
    uint64 expected_l0_born_sequence,
    bool *maintenance_due_out
);

#endif
