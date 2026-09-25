EXTENSION = evoke
MODULE_big = evoke
.DEFAULT_GOAL := all

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs 2>/dev/null)
PG_SHAREDIR := $(shell $(PG_CONFIG) --sharedir 2>/dev/null)
EVOKE_PACKAGED_MODEL_PATH := \
    $(if $(PG_SHAREDIR),$(PG_SHAREDIR),/usr/local/share)/evoke/models/default

ICU_CPPFLAGS := $(shell \
    if pkg-config --exists icu-i18n icu-uc 2>/dev/null; then \
        pkg-config --cflags icu-i18n icu-uc; \
    elif [ -d /opt/homebrew/opt/icu4c@78/include ]; then \
        echo -I/opt/homebrew/opt/icu4c@78/include; \
    elif [ -d /opt/homebrew/opt/icu4c/include ]; then \
        echo -I/opt/homebrew/opt/icu4c/include; \
    else \
        echo; \
    fi)

ICU_LIBS := $(shell \
    if pkg-config --exists icu-i18n icu-uc 2>/dev/null; then \
        pkg-config --libs icu-i18n icu-uc; \
    elif [ -d /opt/homebrew/opt/icu4c@78/lib ]; then \
        echo -L/opt/homebrew/opt/icu4c@78/lib \
            -licui18n -licuuc -licudata; \
    elif [ -d /opt/homebrew/opt/icu4c/lib ]; then \
        echo -L/opt/homebrew/opt/icu4c/lib \
            -licui18n -licuuc -licudata; \
    else \
        echo -licui18n -licuuc -licudata; \
    fi)

OPENSSL_CPPFLAGS := $(shell \
    if pkg-config --exists openssl 2>/dev/null; then \
        pkg-config --cflags openssl; \
    elif [ -d /opt/homebrew/opt/openssl@3/include ]; then \
        echo -I/opt/homebrew/opt/openssl@3/include; \
    elif [ -d /opt/homebrew/opt/openssl/include ]; then \
        echo -I/opt/homebrew/opt/openssl/include; \
    else \
        echo; \
    fi)

OPENSSL_LIBS := $(shell \
    if pkg-config --exists openssl 2>/dev/null; then \
        pkg-config --libs openssl; \
    elif [ -d /opt/homebrew/opt/openssl@3/lib ]; then \
        echo -L/opt/homebrew/opt/openssl@3/lib -lcrypto; \
    elif [ -d /opt/homebrew/opt/openssl/lib ]; then \
        echo -L/opt/homebrew/opt/openssl/lib -lcrypto; \
    else \
        echo -lcrypto; \
    fi)

DL_LIBS := $(shell \
    uname -s 2>/dev/null | grep -q '^Linux' && echo -ldl || true)

HOST_CPU_AVX2_STATUS := $(shell \
    if command -v lscpu >/dev/null 2>&1; then \
        lscpu 2>/dev/null | awk '\
            /Flags:/ { \
                for (i = 2; i <= NF; i++) { \
                    if ($$i == "avx2") { \
                        print "yes"; \
                        found = 1; \
                        exit; \
                    } \
                } \
            } \
            END { \
                if (!found) { \
                    print "no"; \
                } \
            }'; \
    elif command -v sysctl >/dev/null 2>&1; then \
        if sysctl -n machdep.cpu.features machdep.cpu.leaf7_features \
            2>/dev/null | tr '[:upper:]' '[:lower:]' | grep -qw avx2; then \
            echo yes; \
        else \
            echo no; \
        fi; \
    else \
        echo unknown; \
    fi)

ifeq ($(EVOKE_FORCE_SCALAR),1)
PG_CPPFLAGS += -DEVOKE_FORCE_SCALAR=1
$(warning Building evoke with AVX2 dispatch disabled; query scoring \
may be slower on this machine.)
else ifeq ($(HOST_CPU_AVX2_STATUS),no)
$(warning Building evoke on a host without AVX2 support; the \
extension will build and run, but this machine will use the scalar \
scoring path and may be slower.)
else ifeq ($(HOST_CPU_AVX2_STATUS),unknown)
$(warning Could not detect host AVX2 support; evoke will still \
build, but may fall back to the scalar scoring path on this machine.)
endif

OBJS = \
	src/evoke_am.o \
	src/evoke_am_accelerator.o \
    src/evoke_am_build.o \
    src/evoke_am_hot_fold.o \
    src/evoke_am_maintenance.o \
    src/evoke_am_meta.o \
    src/evoke_am_mutation.o \
    src/evoke_am_options.o \
    src/evoke_am_preload.o \
    src/evoke_am_resident_fold.o \
    src/evoke_am_reclamation.o \
    src/evoke_am_scheduler.o \
    src/evoke_am_scan.o \
    src/evoke_am_sql.o \
    src/evoke_am_test_support.o \
    src/evoke_filter.o \
    src/evoke_pg.o \
    src/evoke_pg_common.o \
    src/evoke_planner.o \
    src/evoke_semantic.o \
    src/evoke_semantic_accelerator.o \
    src/evoke_semantic_accelerator_builder.o \
    src/evoke_semantic_accelerator_directory.o \
    src/evoke_semantic_forward.o \
    src/evoke_semantic_forward_bound.o \
	src/evoke_semantic_impact_frontier.o \
	src/evoke_scope.o \
	src/evoke_scope_pg.o \
    src/evoke_weighted_space_saving.o \
    src/evoke_semantic_bmp.o \
    src/evoke_block_ranges.o \
    src/evoke_core.o \
    src/evoke_page_query.o \
    src/evoke_query.o \
    src/evoke_p2_runtime.o \
	src/evoke_document_tid_lookup.o \
    src/evoke_document_cow.o \
    src/evoke_initial_fold_stream.o \
    src/evoke_lexicon_cow.o \
    src/evoke_prefix_cow.o \
    src/evoke_posting_heat.o \
    src/evoke_segment_pages.o \
    src/evoke_segments.o \
    src/evoke_term_cow.o \
    src/evoke_storage.o \
    src/evoke_stem.o
OBJS += src/evoke_text.o

EVOKE_HEADERS := $(wildcard src/*.h)
$(OBJS): $(EVOKE_HEADERS)

src/evoke_am.o src/evoke_semantic.o: \
    src/evoke_semantic.h \
    src/evoke_runtime_service.h

src/evoke_segment_pages.o: \
    src/evoke_segment_pages.h \
    src/evoke_segments.h

src/evoke_document_cow.o: \
    src/evoke_document_cow.h \
    src/evoke_segments.h

src/evoke_lexicon_cow.o: \
    src/evoke_lexicon_cow.h \
    src/evoke_core.h

src/evoke_prefix_cow.o: \
    src/evoke_prefix_cow.h \
    src/evoke_segments.h

src/evoke_term_cow.o: \
    src/evoke_term_cow.h \
    src/evoke_segments.h

EVOKE_EXTENSION_VERSION := $(shell sed -n \
    "s/^default_version = '\(.*\)'$$/\1/p" evoke.control)
DATA = sql/evoke--$(EVOKE_EXTENSION_VERSION).sql
REGRESS = evoke_integration

EVOKE_RUNTIME_SERVER := build/evoke-runtime-server
EVOKE_RUNTIME_SERVER_SRC := src/evoke_runtime_server.cc
EVOKE_RUNTIME_SERVER_CXXFLAGS ?= \
    -std=c++17 -Wall -Wextra -Wpedantic -O2
EVOKE_RUNTIME_SERVER_CPPFLAGS := \
    $(shell pkg-config --cflags jansson 2>/dev/null) \
    $(ICU_CPPFLAGS) \
    $(OPENSSL_CPPFLAGS) \
    $(shell pkg-config --cflags libonnxruntime 2>/dev/null) \
    -DEVOKE_PACKAGED_MODEL_PATH='"$(EVOKE_PACKAGED_MODEL_PATH)"'
EVOKE_RUNTIME_SERVER_LDFLAGS :=
EVOKE_RUNTIME_SERVER_LIBS := \
    $(shell pkg-config --libs jansson 2>/dev/null || echo -ljansson) \
    $(ICU_LIBS) \
    $(OPENSSL_LIBS) \
    $(shell pkg-config --libs libonnxruntime 2>/dev/null || \
        echo -lonnxruntime) \
    -pthread

.PHONY: runtime-server
runtime-server: $(EVOKE_RUNTIME_SERVER)

$(EVOKE_RUNTIME_SERVER): $(EVOKE_RUNTIME_SERVER_SRC)
	mkdir -p $(dir $@)
	$(CXX) $(EVOKE_RUNTIME_SERVER_CXXFLAGS) \
	    $(EVOKE_RUNTIME_SERVER_CPPFLAGS) \
	    -o $@ $< \
	    $(EVOKE_RUNTIME_SERVER_LDFLAGS) \
	    $(EVOKE_RUNTIME_SERVER_LIBS)

PG_CPPFLAGS += \
    -I$(CURDIR)/src \
    $(ICU_CPPFLAGS) \
    -DEVOKE_PACKAGED_MODEL_PATH='"$(EVOKE_PACKAGED_MODEL_PATH)"'
SHLIB_LINK += $(ICU_LIBS) $(DL_LIBS)

EVOKE_ENABLE_ONNXRUNTIME ?= 1
ifneq ($(EVOKE_ENABLE_ONNXRUNTIME),0)
ifneq ($(EVOKE_ENABLE_ONNXRUNTIME),1)
$(error EVOKE_ENABLE_ONNXRUNTIME must be 0 or 1)
endif
endif

EVOKE_ONNXRUNTIME_REQUIRED_VERSION := $(shell tr -d '[:space:]' \
    < packaging/onnxruntime.version)

EVOKE_CLEAN_ONLY_GOALS := clean clean-evoke-runtime-server
EVOKE_BUILD_GOALS := $(if $(MAKECMDGOALS),\
    $(filter-out $(EVOKE_CLEAN_ONLY_GOALS),$(MAKECMDGOALS)),default)

ifneq ($(strip $(EVOKE_BUILD_GOALS)),)
ifeq ($(EVOKE_ENABLE_ONNXRUNTIME),1)
ONNXRUNTIME_CPPFLAGS := $(shell pkg-config --cflags libonnxruntime 2>/dev/null)
ONNXRUNTIME_LIBS := $(shell pkg-config --libs libonnxruntime 2>/dev/null)
ONNXRUNTIME_VERSION := $(shell pkg-config --modversion libonnxruntime 2>/dev/null)
ONNXRUNTIME_LIB_DIRS := $(filter -L%,$(ONNXRUNTIME_LIBS))
ONNXRUNTIME_LINK_LIBS := $(filter-out -L%,$(ONNXRUNTIME_LIBS))
HOST_OS := $(shell uname -s)
ifeq ($(strip $(ONNXRUNTIME_CPPFLAGS)),)
$(error EVOKE_ENABLE_ONNXRUNTIME=1 requires pkg-config libonnxruntime)
endif
ifneq ($(ONNXRUNTIME_VERSION),$(EVOKE_ONNXRUNTIME_REQUIRED_VERSION))
$(error Evoke requires ONNX Runtime $(EVOKE_ONNXRUNTIME_REQUIRED_VERSION), \
    but pkg-config resolved $(ONNXRUNTIME_VERSION))
endif
PG_CPPFLAGS += -DEVOKE_ENABLE_ONNXRUNTIME=1 $(ONNXRUNTIME_CPPFLAGS)
# Put ONNX Runtime library directories before PostgreSQL's default linker
# search path. Some distributions ship a CPU-only libonnxruntime under
# /usr/lib; accelerator builds must not accidentally bind that library when a
# CUDA/CoreML package is supplied through pkg-config.
PG_LDFLAGS += $(ONNXRUNTIME_LIB_DIRS)
ifeq ($(HOST_OS),Darwin)
PG_LDFLAGS += -Wl,-rpath,@loader_path
else
# Release packages place libonnxruntime beside the extension in pkglibdir.
PG_LDFLAGS += -Wl,-rpath,\$$ORIGIN
endif
SHLIB_LINK += $(ONNXRUNTIME_LINK_LIBS)
endif
endif

include $(PGXS)

.PHONY: clean-evoke-runtime-server
clean: clean-evoke-runtime-server

clean-evoke-runtime-server:
	rm -f $(EVOKE_RUNTIME_SERVER)
