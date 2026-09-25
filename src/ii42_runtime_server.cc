#include <arpa/inet.h>
#include <errno.h>
#include <jansson.h>
#include <netinet/in.h>
#include <onnxruntime_c_api.h>
#include <openssl/evp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unicode/uregex.h>
#include <unicode/ustring.h>
#include <unicode/utf16.h>

namespace {

constexpr int DEFAULT_PORT = 8042;
constexpr int DEFAULT_MAX_BATCH_SIZE = 128;
constexpr int MAX_BATCH_SIZE = 512;
constexpr int DEFAULT_WORKER_COUNT = 1;
constexpr int MAX_WORKER_COUNT = 128;
constexpr int MAX_QUEUE_DEPTH = 1024;
constexpr int DEFAULT_CONNECTION_IDLE_TIMEOUT_S = 30;
constexpr int MAX_CONNECTIONS = 4096;
constexpr int LISTEN_BACKLOG = 1024;
constexpr size_t DEFAULT_MAX_REQUEST_BYTES = 8U << 20;
constexpr size_t RUNTIME_TRANSPORT_MAX_BYTES = 1U << 20;
constexpr const char *JSON_CONTENT_TYPE = "application/json";
constexpr const char *RUNTIME_ABI = "evoke_p2_unified_text_atoms_v2";
constexpr const char *BPE_PATTERN =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|"
    " ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
constexpr int BOS_TOKEN_ID = 0;
constexpr int EOS_TOKEN_ID = 2;
constexpr int DEFAULT_WINDOW_OVERLAP = 64;
constexpr int DEFAULT_MAX_WINDOWS = 64;
constexpr int MAX_RUNTIME_WINDOWS = 256;
constexpr size_t SHA256_HEX_LENGTH = 64;

volatile sig_atomic_t stop_requested = 0;
int listen_fd = -1;

struct Options {
    std::string model_path;
    std::string checkout_signature;
    std::string runtime_precision = "fp16";
    std::string host = "127.0.0.1";
    int port = DEFAULT_PORT;
    int max_batch_size = DEFAULT_MAX_BATCH_SIZE;
    int max_delay_ms = 5;
    int request_timeout_s = 300;
    int worker_count = DEFAULT_WORKER_COUNT;
    int max_queue_depth = 0;
    int max_connections = 0;
    int connection_idle_timeout_s = DEFAULT_CONNECTION_IDLE_TIMEOUT_S;
    size_t max_request_bytes = DEFAULT_MAX_REQUEST_BYTES;
    int intra_op_threads = 0;
};

struct HttpError : public std::runtime_error {
    int status;
    std::string detail;

    HttpError(int status_code, const std::string &message)
        : std::runtime_error(message), status(status_code)
    {
    }

    HttpError(
        int status_code,
        const std::string &message,
        const std::string &detail_message
    )
        : std::runtime_error(message),
          status(status_code),
          detail(detail_message)
    {
    }
};

class JsonPtr {
public:
    JsonPtr() = default;

    explicit JsonPtr(json_t *value) : value_(value)
    {
    }

    ~JsonPtr()
    {
        json_decref(value_);
    }

    JsonPtr(const JsonPtr &) = delete;
    JsonPtr &operator=(const JsonPtr &) = delete;

    JsonPtr(JsonPtr &&other) noexcept : value_(other.value_)
    {
        other.value_ = nullptr;
    }

    JsonPtr &operator=(JsonPtr &&other) noexcept
    {
        if (this != &other) {
            json_decref(value_);
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }

    json_t *get() const
    {
        return value_;
    }

    json_t *release()
    {
        json_t *value = value_;
        value_ = nullptr;
        return value;
    }

private:
    json_t *value_ = nullptr;
};

std::string dump_json(json_t *root)
{
    char *raw = json_dumps(root, JSON_COMPACT);
    if (raw == nullptr) {
        throw std::runtime_error("could not serialize JSON response");
    }
    std::string result(raw);
    free(raw);
    return result;
}

JsonPtr parse_json_body(const std::string &body)
{
    json_error_t error;
    json_t *root = json_loadb(body.data(), body.size(), 0, &error);
    if (root == nullptr) {
        throw HttpError(400, "invalid JSON body", error.text);
    }
    if (!json_is_object(root)) {
        json_decref(root);
        throw HttpError(400, "JSON body must be an object");
    }
    return JsonPtr(root);
}

void json_set_string(json_t *object, const char *key, const std::string &value)
{
    json_t *string_value = json_string(value.c_str());
    if (string_value == nullptr) {
        throw std::runtime_error("could not construct JSON string");
    }
    json_object_set_new(object, key, string_value);
}

std::string json_string_default(
    json_t *object,
    const char *key,
    const std::string &default_value
)
{
    const char *value = json_string_value(json_object_get(object, key));

    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value;
}

std::string runtime_provider_string_default(
    json_t *runtime_envelope,
    const char *key,
    const std::string &default_value
)
{
    json_t *provider = json_object_get(runtime_envelope, "provider");

    if (!json_is_object(provider)) {
        return default_value;
    }
    return json_string_default(provider, key, default_value);
}

void json_copy_field(json_t *target, json_t *source, const char *key)
{
    json_t *value = json_object_get(source, key);

    if (value != nullptr) {
        json_object_set(target, key, value);
    }
}

std::string json_error_body(
    const std::string &message,
    const std::string &detail = ""
)
{
    JsonPtr root(json_object());
    json_set_string(root.get(), "error", message);
    if (!detail.empty()) {
        json_set_string(root.get(), "detail", detail);
    }
    return dump_json(root.get());
}

void check_json_alloc(json_t *value)
{
    if (value == nullptr) {
        throw std::runtime_error("could not allocate JSON value");
    }
}

struct ModelInfo {
    std::string model_path;
    std::string model_id;
    std::string checkout_signature;
    std::string runtime_precision;
    std::string runtime_abi;
    std::string runtime_output_json = "{}";
    JsonPtr runtime_output;
    std::string provider;
    std::string active_provider;
    std::string document_encoder_path;
    std::string tokenizer_vocabulary_path;
    std::string tokenizer_merges_path;
    std::string input_ids_name;
    std::string attention_mask_name;
    std::string semantic_ids_name;
    std::string semantic_weights_name;
    int max_length = 512;
    int window_stride = 446;
    int max_windows = DEFAULT_MAX_WINDOWS;
    int max_output_atoms = 512;
    int semantic_max_atoms = 192;
    int lexical_dims = 0;
    int total_dims = 0;
};

std::string normalize_runtime_precision(const std::string &value)
{
    std::string normalized = value.empty() ? "fp16" : value;

    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        }
    );
    if (normalized != "fp16" && normalized != "fp32") {
        throw std::runtime_error(
            "runtime precision must be one of fp16 or fp32"
        );
    }
    return normalized;
}

std::string trim_copy(std::string value)
{
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string read_file(const std::string &path, size_t max_bytes)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("could not stat " + path);
    }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0 ||
        static_cast<uint64_t>(st.st_size) > max_bytes) {
        throw std::runtime_error("invalid file size for " + path);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open " + path);
    }
    std::string contents;
    contents.resize(static_cast<size_t>(st.st_size));
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!input) {
        throw std::runtime_error("could not read " + path);
    }
    return contents;
}

JsonPtr parse_json_text(const std::string &text, const std::string &label)
{
    json_error_t error;
    json_t *root = json_loadb(text.data(), text.size(), 0, &error);
    if (root == nullptr || !json_is_object(root)) {
        json_decref(root);
        throw std::runtime_error(label + " is not a JSON object: " +
                                 error.text);
    }
    return JsonPtr(root);
}

JsonPtr read_json_file(const std::string &path, size_t max_bytes)
{
    return parse_json_text(read_file(path, max_bytes), path);
}

bool is_hex_sha256(const std::string &value)
{
    if (value.size() != SHA256_HEX_LENGTH) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isxdigit(ch) != 0;
    });
}

std::string sha256_file(const std::string &path)
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned char buffer[1 << 16];
    unsigned int digest_len = 0;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        throw std::runtime_error("could not open " + path);
    }
    if (ctx == nullptr ||
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("could not initialize SHA256 context");
    }
    while (input) {
        input.read(
            reinterpret_cast<char *>(buffer),
            static_cast<std::streamsize>(sizeof(buffer))
        );
        std::streamsize got = input.gcount();
        if (got > 0) {
            if (EVP_DigestUpdate(
                    ctx,
                    buffer,
                    static_cast<size_t>(got)
                ) != 1) {
                EVP_MD_CTX_free(ctx);
                throw std::runtime_error("could not update SHA256 context");
            }
        }
    }
    if (!input.eof()) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("could not read " + path);
    }
    if (EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1 ||
        digest_len != SHA256_HEX_LENGTH / 2) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("could not finish SHA256 digest");
    }
    EVP_MD_CTX_free(ctx);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_len; i++) {
        out << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return out.str();
}

std::string join_path(const std::string &base, const std::string &relpath)
{
    if (base.empty() || base.front() != '/') {
        throw std::runtime_error("model path must be absolute");
    }
    if (relpath.empty() || relpath.front() == '/' ||
        relpath.find("..") != std::string::npos ||
        relpath.find('\n') != std::string::npos ||
        relpath.find('\r') != std::string::npos) {
        throw std::runtime_error("unsafe model artifact path: " + relpath);
    }
    if (base.back() == '/') {
        return base + relpath;
    }
    return base + "/" + relpath;
}

std::string json_required_string(json_t *object, const char *key)
{
    const char *value = json_string_value(json_object_get(object, key));
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("missing string key ") + key);
    }
    return value;
}

int json_int_key(json_t *object, const char *key, int fallback,
                 int min_value, int max_value)
{
    json_t *value = json_object_get(object, key);
    long long parsed = fallback;

    if (json_is_integer(value)) {
        parsed = json_integer_value(value);
    }
    if (parsed < min_value || parsed > max_value) {
        throw std::runtime_error(std::string("invalid integer key ") + key);
    }
    return static_cast<int>(parsed);
}

int json_int_path(json_t *object, const char *first, const char *second,
                  int fallback, int min_value, int max_value)
{
    json_t *child = json_object_get(object, first);
    if (!json_is_object(child)) {
        return fallback;
    }
    return json_int_key(child, second, fallback, min_value, max_value);
}

std::string artifact_relpath(json_t *manifest, const char *artifact_key)
{
    json_t *artifacts = json_object_get(manifest, "artifacts");
    json_t *artifact = json_is_object(artifacts)
        ? json_object_get(artifacts, artifact_key)
        : nullptr;
    if (!json_is_object(artifact)) {
        throw std::runtime_error(
            std::string("missing model artifact ") + artifact_key
        );
    }
    return json_required_string(artifact, "path");
}

std::string artifact_path(json_t *manifest, const std::string &model_path,
                          const char *artifact_key)
{
    return join_path(model_path, artifact_relpath(manifest, artifact_key));
}

void verify_artifact_sha256(json_t *manifest, const std::string &model_path,
                            const char *artifact_key)
{
    json_t *artifact = json_object_get(
        json_object_get(manifest, "artifacts"),
        artifact_key
    );
    std::string expected = json_required_string(artifact, "sha256");
    if (!is_hex_sha256(expected)) {
        throw std::runtime_error(
            std::string("invalid sha256 for artifact ") + artifact_key
        );
    }
    std::string path = artifact_path(manifest, model_path, artifact_key);
    std::string actual = sha256_file(path);
    if (actual != expected) {
        throw std::runtime_error(
            std::string("artifact sha256 mismatch for ") + artifact_key +
            ": expected " + expected + " got " + actual
        );
    }
}

const OrtApi *ort_api()
{
    if (std::getenv("ORT_DISABLE_TELEMETRY") == nullptr) {
        setenv("ORT_DISABLE_TELEMETRY", "1", 0);
    }
    const OrtApiBase *api_base = OrtGetApiBase();
    if (api_base == nullptr) {
        throw std::runtime_error("ONNX Runtime API base is unavailable");
    }
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        throw std::runtime_error("ONNX Runtime API is unavailable");
    }
    return api;
}

void ort_check(const OrtApi *ort, OrtStatus *status, const char *step)
{
    if (status == nullptr) {
        return;
    }
    const char *message = ort->GetErrorMessage(status);
    std::string detail = message == nullptr ? "unknown error" : message;
    ort->ReleaseStatus(status);
    throw std::runtime_error(std::string(step) + ": " + detail);
}

bool ort_provider_available(const OrtApi *ort, const char *provider)
{
    OrtStatus *status;
    char **providers = nullptr;
    int provider_count = 0;
    bool available = false;

    status = ort->GetAvailableProviders(&providers, &provider_count);
    if (status != nullptr) {
        ort->ReleaseStatus(status);
        return false;
    }
    for (int i = 0; i < provider_count; i++) {
        if (std::strcmp(providers[i], provider) == 0) {
            available = true;
            break;
        }
    }
    status = ort->ReleaseAvailableProviders(providers, provider_count);
    if (status != nullptr) {
        ort->ReleaseStatus(status);
    }
    return available;
}

std::vector<std::string> provider_candidates(
    const OrtApi *ort,
    const std::string &precision
)
{
    std::vector<std::string> providers;
    auto add_if_available = [&](const char *ort_name, const char *name) {
        if (ort_provider_available(ort, ort_name)) {
            providers.emplace_back(name);
        }
    };

    if (precision == "fp16") {
        add_if_available("TensorrtExecutionProvider", "tensorrt");
        add_if_available("CoreMLExecutionProvider", "coreml");
        add_if_available("CUDAExecutionProvider", "cuda");
    } else {
        add_if_available("CUDAExecutionProvider", "cuda");
        add_if_available("TensorrtExecutionProvider", "tensorrt");
        add_if_available("CoreMLExecutionProvider", "coreml");
    }
    providers.emplace_back("cpu");
    return providers;
}

std::string active_provider(const OrtApi *ort, const std::string &precision)
{
    return provider_candidates(ort, precision).front();
}

std::string load_checkout_signature(const Options &options,
                                    const std::string &model_path)
{
    if (!options.checkout_signature.empty()) {
        return options.checkout_signature;
    }
    const char *env = std::getenv("II42_CHECKOUT_SIGNATURE");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    std::string sidecar_path = join_path(model_path, "checkout_signature");
    struct stat st;
    if (stat(sidecar_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
        return trim_copy(read_file(sidecar_path, 4096));
    }
    throw std::runtime_error(
        "direct runtime server requires --checkout-signature or "
        "II42_CHECKOUT_SIGNATURE"
    );
}

ModelInfo load_model(const Options &options)
{
    ModelInfo model;
#ifdef II42_PACKAGED_MODEL_PATH
    std::string path = options.model_path.empty()
        ? II42_PACKAGED_MODEL_PATH
        : options.model_path;
#else
    std::string path = options.model_path;
#endif
    if (path.empty()) {
        throw std::runtime_error("direct runtime server requires --model-path");
    }
    std::string manifest_path = join_path(path, "manifest.json");
    JsonPtr manifest = read_json_file(manifest_path, 16U << 20);

    if (json_string_default(manifest.get(), "runtime", "") !=
            "onnxruntime" ||
        json_string_default(manifest.get(), "model_format", "") != "onnx" ||
        json_string_default(manifest.get(), "runtime_abi", "") !=
            RUNTIME_ABI) {
        throw std::runtime_error("unsupported ii42 model manifest");
    }

    model.model_path = path;
    model.model_id = json_required_string(manifest.get(), "model_id");
    model.checkout_signature = load_checkout_signature(options, path);
    if (!is_hex_sha256(model.checkout_signature)) {
        throw std::runtime_error("checkout signature must be 64 hex chars");
    }
    model.runtime_precision = normalize_runtime_precision(
        options.runtime_precision
    );
    model.runtime_abi = json_required_string(manifest.get(), "runtime_abi");
    model.provider = "auto";
    model.active_provider = active_provider(ort_api(), model.runtime_precision);
    model.document_encoder_path =
        artifact_path(manifest.get(), path, "document_encoder");
    model.tokenizer_vocabulary_path =
        artifact_path(manifest.get(), path, "tokenizer_vocabulary");
    model.tokenizer_merges_path =
        artifact_path(manifest.get(), path, "tokenizer_merges");

    const char *required_artifacts[] = {
        "document_encoder",
        "tokenizer_vocabulary",
        "tokenizer_merges",
        "semantic_runtime",
        "atom_space"
    };
    for (const char *artifact : required_artifacts) {
        verify_artifact_sha256(manifest.get(), path, artifact);
    }

    JsonPtr semantic_runtime = read_json_file(
        artifact_path(manifest.get(), path, "semantic_runtime"),
        64U << 20
    );
    JsonPtr atom_space = read_json_file(
        artifact_path(manifest.get(), path, "atom_space"),
        64U << 20
    );

    model.max_length = json_int_key(
        semantic_runtime.get(),
        "max_length",
        512,
        3,
        4096
    );
    int window_overlap = json_int_key(
        semantic_runtime.get(),
        "window_overlap",
        std::min(DEFAULT_WINDOW_OVERLAP, model.max_length - 3),
        0,
        model.max_length - 3
    );
    model.window_stride = model.max_length - 2 - window_overlap;
    model.max_windows = json_int_key(
        semantic_runtime.get(),
        "max_windows",
        DEFAULT_MAX_WINDOWS,
        1,
        MAX_RUNTIME_WINDOWS
    );

    json_t *runtime_output = json_object_get(manifest.get(), "runtime_output");
    if (!json_is_object(runtime_output)) {
        throw std::runtime_error("manifest runtime_output must be an object");
    }
    model.runtime_output = JsonPtr(json_deep_copy(runtime_output));
    check_json_alloc(model.runtime_output.get());
    model.runtime_output_json = dump_json(model.runtime_output.get());
    model.max_output_atoms = json_int_key(
        runtime_output,
        "max_atoms",
        512,
        1,
        65536
    );
    model.semantic_max_atoms = json_int_key(
        runtime_output,
        "document_semantic_max_atoms",
        192,
        1,
        65536
    );
    model.lexical_dims = json_int_path(
        atom_space.get(),
        "lexical",
        "end_exclusive",
        0,
        1,
        std::numeric_limits<int>::max()
    );
    model.total_dims = json_int_key(
        atom_space.get(),
        "latent_dims",
        0,
        model.lexical_dims + 1,
        std::numeric_limits<int>::max()
    );
    int manifest_dims = json_int_key(
        manifest.get(),
        "latent_dims",
        0,
        model.lexical_dims + 1,
        std::numeric_limits<int>::max()
    );
    if (manifest_dims != model.total_dims) {
        throw std::runtime_error("manifest and atom_space dimensions differ");
    }

    json_t *runtime_io = json_object_get(manifest.get(), "runtime_io");
    if (!json_is_object(runtime_io)) {
        throw std::runtime_error("manifest runtime_io must be an object");
    }
    model.input_ids_name = json_required_string(runtime_io, "input_ids");
    model.attention_mask_name =
        json_required_string(runtime_io, "attention_mask");
    model.semantic_ids_name = json_required_string(runtime_io, "semantic_ids");
    model.semantic_weights_name =
        json_required_string(runtime_io, "semantic_weights");
    return model;
}

std::string pair_key(const std::string &left, const std::string &right)
{
    std::string key = left;
    key.push_back('\0');
    key.append(right);
    return key;
}

class RobertaTokenizer {
public:
    RobertaTokenizer(
        const std::string &vocabulary_path,
        const std::string &merges_path
    )
    {
        load_vocabulary(vocabulary_path);
        load_merges(merges_path);
        build_byte_symbols();

        UErrorCode status = U_ZERO_ERROR;
        pattern_ = uregex_openC(BPE_PATTERN, 0, nullptr, &status);
        if (U_FAILURE(status) || pattern_ == nullptr) {
            throw std::runtime_error("could not compile BPE tokenizer pattern");
        }
    }

    ~RobertaTokenizer()
    {
        if (pattern_ != nullptr) {
            uregex_close(pattern_);
        }
    }

    RobertaTokenizer(const RobertaTokenizer &) = delete;
    RobertaTokenizer &operator=(const RobertaTokenizer &) = delete;

    struct Input {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        int token_count = 0;
    };

    struct Windows {
        std::vector<Input> windows;
        int full_token_count = 0;
        int window_stride = 0;
    };

    Windows tokenize_windows(
        const std::string &text,
        int max_length,
        int window_stride,
        int max_windows
    ) const
    {
        if (max_length < 3 || max_windows < 1) {
            throw std::runtime_error("invalid tokenizer window settings");
        }
        int window_content_capacity = max_length - 2;
        if (window_stride < 1 || window_stride > window_content_capacity) {
            throw std::runtime_error("invalid tokenizer window stride");
        }
        int content_limit = window_content_capacity +
            window_stride * std::max(max_windows - 1, 0);
        std::vector<int64_t> tokens;
        std::vector<UChar> utf16 = utf8_to_utf16(text);
        UErrorCode status = U_ZERO_ERROR;
        URegularExpression *pattern = uregex_clone(pattern_, &status);
        if (U_FAILURE(status) || pattern == nullptr) {
            throw std::runtime_error("could not clone BPE tokenizer pattern");
        }

        try {
            uregex_setText(
                pattern,
                utf16.data(),
                static_cast<int32_t>(utf16.size()),
                &status
            );
            while (U_SUCCESS(status) &&
                   uregex_findNext(pattern, &status)) {
                int32_t start = uregex_start(pattern, 0, &status);
                int32_t end = uregex_end(pattern, 0, &status);
                if (U_FAILURE(status)) {
                    break;
                }
                if (!append_piece(
                    utf16_slice_to_utf8(utf16, start, end),
                    content_limit,
                    tokens
                )) {
                    break;
                }
            }
            uregex_close(pattern);
        } catch (...) {
            uregex_close(pattern);
            throw;
        }
        if (U_FAILURE(status)) {
            throw std::runtime_error("BPE tokenization failed");
        }

        Windows output;
        output.full_token_count = static_cast<int>(tokens.size()) + 2;
        output.window_stride = window_stride;

        int window_start = 0;
        do {
            int remaining = static_cast<int>(tokens.size()) - window_start;
            int content_count = std::min(remaining, window_content_capacity);
            int token_count = content_count + 2;
            Input input;

            input.input_ids.reserve(static_cast<size_t>(token_count));
            input.attention_mask.assign(static_cast<size_t>(token_count), 1);
            input.input_ids.push_back(BOS_TOKEN_ID);
            for (int i = 0; i < content_count; i++) {
                input.input_ids.push_back(tokens[window_start + i]);
            }
            input.input_ids.push_back(EOS_TOKEN_ID);
            input.token_count = token_count;
            output.windows.push_back(std::move(input));
            if (window_start + content_count >=
                static_cast<int>(tokens.size())) {
                break;
            }
            window_start += window_stride;
        } while (static_cast<int>(output.windows.size()) < max_windows);

        return output;
    }

private:
    void load_vocabulary(const std::string &path)
    {
        JsonPtr root = read_json_file(path, 64U << 20);
        const char *key;
        json_t *value;

        json_object_foreach(root.get(), key, value) {
            if (!json_is_integer(value)) {
                throw std::runtime_error("invalid tokenizer vocabulary");
            }
            vocabulary_.emplace(key, static_cast<int>(json_integer_value(value)));
        }
        if (vocabulary_.empty()) {
            throw std::runtime_error("tokenizer vocabulary is empty");
        }
    }

    void load_merges(const std::string &path)
    {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("could not open tokenizer merges");
        }
        std::string line;
        int rank = 0;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty() || line.front() == '#') {
                continue;
            }
            size_t separator = line.find(' ');
            if (separator == std::string::npos ||
                separator == 0 ||
                separator + 1 >= line.size()) {
                throw std::runtime_error("invalid tokenizer merge row");
            }
            merge_ranks_.emplace(
                pair_key(line.substr(0, separator),
                         line.substr(separator + 1)),
                rank++
            );
        }
        if (merge_ranks_.empty()) {
            throw std::runtime_error("tokenizer merges are empty");
        }
    }

    static bool identity_codepoint(unsigned int value)
    {
        return (value >= 33 && value <= 126) ||
               (value >= 161 && value <= 172) ||
               (value >= 174 && value <= 255);
    }

    static int byte_codepoint(unsigned int value)
    {
        if (identity_codepoint(value)) {
            return static_cast<int>(value);
        }
        int extra = 0;
        for (unsigned int candidate = 0; candidate < value; candidate++) {
            if (!identity_codepoint(candidate)) {
                extra++;
            }
        }
        return 256 + extra;
    }

    static std::string codepoint_to_utf8(int codepoint)
    {
        UChar unicode[2];
        int32_t unicode_length = 0;
        int32_t utf8_length = 0;
        UErrorCode status = U_ZERO_ERROR;

        U16_APPEND_UNSAFE(unicode, unicode_length, codepoint);
        u_strToUTF8(nullptr, 0, &utf8_length, unicode, unicode_length, &status);
        if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
            throw std::runtime_error("could not size byte symbol");
        }
        status = U_ZERO_ERROR;
        std::string output(static_cast<size_t>(utf8_length) + 1, '\0');
        u_strToUTF8(
            output.data(),
            utf8_length + 1,
            nullptr,
            unicode,
            unicode_length,
            &status
        );
        if (U_FAILURE(status)) {
            throw std::runtime_error("could not encode byte symbol");
        }
        output.resize(static_cast<size_t>(utf8_length));
        return output;
    }

    void build_byte_symbols()
    {
        byte_symbols_.resize(256);
        for (int i = 0; i < 256; i++) {
            byte_symbols_[static_cast<size_t>(i)] =
                codepoint_to_utf8(byte_codepoint(static_cast<unsigned int>(i)));
        }
    }

    static std::vector<UChar> utf8_to_utf16(const std::string &text)
    {
        UErrorCode status = U_ZERO_ERROR;
        int32_t length = 0;

        u_strFromUTF8(nullptr, 0, &length, text.c_str(), -1, &status);
        if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
            throw std::runtime_error("text is not valid UTF-8");
        }
        status = U_ZERO_ERROR;
        std::vector<UChar> output(static_cast<size_t>(length) + 1);
        u_strFromUTF8(output.data(), length + 1, nullptr,
                      text.c_str(), -1, &status);
        if (U_FAILURE(status)) {
            throw std::runtime_error("could not convert text to UTF-16");
        }
        output.resize(static_cast<size_t>(length));
        return output;
    }

    static std::string utf16_slice_to_utf8(
        const std::vector<UChar> &text,
        int32_t start,
        int32_t end
    )
    {
        UErrorCode status = U_ZERO_ERROR;
        int32_t length = 0;

        u_strToUTF8(
            nullptr,
            0,
            &length,
            text.data() + start,
            end - start,
            &status
        );
        if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
            throw std::runtime_error("could not size UTF-8 token");
        }
        status = U_ZERO_ERROR;
        std::string output(static_cast<size_t>(length) + 1, '\0');
        u_strToUTF8(
            output.data(),
            length + 1,
            nullptr,
            text.data() + start,
            end - start,
            &status
        );
        if (U_FAILURE(status)) {
            throw std::runtime_error("could not encode UTF-8 token");
        }
        output.resize(static_cast<size_t>(length));
        return output;
    }

    bool append_piece(
        const std::string &piece,
        int content_limit,
        std::vector<int64_t> &tokens
    ) const
    {
        std::vector<std::string> symbols;
        symbols.reserve(piece.size());
        for (unsigned char byte : piece) {
            symbols.push_back(byte_symbols_[byte]);
        }

        while (symbols.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            std::string best_key;
            bool has_pair = false;

            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                std::string key = pair_key(symbols[i], symbols[i + 1]);
                auto found = merge_ranks_.find(key);
                if (found != merge_ranks_.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best_key = std::move(key);
                    has_pair = true;
                }
            }
            if (!has_pair) {
                break;
            }
            std::vector<std::string> next;
            next.reserve(symbols.size());
            for (size_t i = 0; i < symbols.size();) {
                bool merge = i + 1 < symbols.size() &&
                    pair_key(symbols[i], symbols[i + 1]) == best_key;
                if (merge) {
                    next.push_back(symbols[i] + symbols[i + 1]);
                    i += 2;
                } else {
                    next.push_back(std::move(symbols[i]));
                    i++;
                }
            }
            symbols = std::move(next);
        }

        for (const std::string &symbol : symbols) {
            auto found = vocabulary_.find(symbol);
            if (found == vocabulary_.end()) {
                throw std::runtime_error("BPE token is absent from vocabulary");
            }
            if (static_cast<int>(tokens.size()) >= content_limit) {
                return false;
            }
            tokens.push_back(found->second);
        }
        return true;
    }

    std::unordered_map<std::string, int> vocabulary_;
    std::unordered_map<std::string, int> merge_ranks_;
    std::vector<std::string> byte_symbols_;
    URegularExpression *pattern_ = nullptr;
};

struct CompiledResult {
    std::vector<int32_t> atom_ids;
    std::vector<float> atom_weights;
    int token_count = 0;
    int window_count = 0;
    int window_stride = 0;
    double semantic_proxy = 0.0;
};

class DirectRuntimeEngine {
public:
    DirectRuntimeEngine(ModelInfo &model, const Options &options)
        : model_(model),
          api_(ort_api()),
          tokenizer_(model.tokenizer_vocabulary_path,
                     model.tokenizer_merges_path)
    {
        double started = monotonic_ms();

        ort_check(api_, api_->CreateEnv(
            ORT_LOGGING_LEVEL_WARNING,
            "ii42-runtime-server",
            &env_
        ), "CreateEnv");
        create_session(options);
        session_load_ms_ = monotonic_ms() - started;
    }

    ~DirectRuntimeEngine()
    {
        if (session_ != nullptr) {
            api_->ReleaseSession(session_);
        }
        if (session_options_ != nullptr) {
            api_->ReleaseSessionOptions(session_options_);
        }
        if (env_ != nullptr) {
            api_->ReleaseEnv(env_);
        }
    }

    DirectRuntimeEngine(const DirectRuntimeEngine &) = delete;
    DirectRuntimeEngine &operator=(const DirectRuntimeEngine &) = delete;

    std::string encode(
        const std::string &mode,
        const std::vector<std::string> &texts,
        int max_batch_size
    )
    {
        if (mode != "document") {
            throw HttpError(
                400,
                "unsupported encode mode",
                "direct accelerator currently supports document mode"
            );
        }
        double started = monotonic_ms();
        std::vector<CompiledResult> compiled =
            encode_documents(texts, max_batch_size);
        double elapsed = monotonic_ms() - started;

        JsonPtr root(json_object());
        check_json_alloc(root.get());
        json_t *results = json_array();
        check_json_alloc(results);
        int total_windows = 0;
        int sequence_length = 0;

        for (const CompiledResult &result : compiled) {
            total_windows += result.window_count;
            sequence_length = std::max(sequence_length, result.token_count);
            json_array_append_new(results, compiled_result_json(result));
        }
        json_object_set_new(root.get(), "results", results);
        json_set_string(root.get(), "runtime", "onnxruntime");
        json_set_string(root.get(), "runtime_abi", model_.runtime_abi);
        json_set_string(
            root.get(),
            "checkout_signature",
            model_.checkout_signature
        );
        json_set_string(root.get(), "compiler_role", "document");
        JsonPtr provider(json_object());
        json_set_string(provider.get(), "requested", "auto");
        json_set_string(provider.get(), "active", model_.active_provider);
        json_object_set_new(root.get(), "provider", provider.release());
        json_set_string(
            root.get(),
            "runtime_precision",
            model_.runtime_precision
        );
        json_object_set_new(
            root.get(),
            "batch_size",
            json_integer(static_cast<json_int_t>(texts.size()))
        );
        json_object_set_new(
            root.get(),
            "sequence_length",
            json_integer(sequence_length)
        );
        json_object_set_new(
            root.get(),
            "window_count",
            json_integer(total_windows)
        );
        json_object_set_new(root.get(), "latency_ms", json_real(elapsed));
        json_object_set_new(root.get(), "session_cache_hit", json_true());
        json_object_set_new(
            root.get(),
            "session_load_ms",
            json_real(session_load_ms_)
        );
        return dump_json(root.get());
    }

private:
    struct WindowRef {
        size_t row;
        const RobertaTokenizer::Input *input;
    };

    static double monotonic_ms()
    {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::milli>(now).count();
    }

    void create_session(const Options &options)
    {
        std::ostringstream errors;
        std::vector<std::string> providers =
            provider_candidates(api_, model_.runtime_precision);

        for (const std::string &provider : providers) {
            OrtSessionOptions *candidate_options = nullptr;
            OrtSession *candidate_session = nullptr;

            try {
                ort_check(api_, api_->CreateSessionOptions(
                    &candidate_options
                ), "CreateSessionOptions");
                if (options.intra_op_threads > 0) {
                    ort_check(api_, api_->SetIntraOpNumThreads(
                        candidate_options,
                        options.intra_op_threads
                    ), "SetIntraOpNumThreads");
                }
                configure_provider(provider, candidate_options);
                ort_check(api_, api_->CreateSession(
                    env_,
                    model_.document_encoder_path.c_str(),
                    candidate_options,
                    &candidate_session
                ), "CreateSession(document)");
                session_options_ = candidate_options;
                session_ = candidate_session;
                model_.active_provider = provider;
                return;
            } catch (const std::exception &ex) {
                if (candidate_session != nullptr) {
                    api_->ReleaseSession(candidate_session);
                }
                if (candidate_options != nullptr) {
                    api_->ReleaseSessionOptions(candidate_options);
                }
                std::cerr
                    << "ii42-runtime-server provider " << provider
                    << " unavailable: " << ex.what() << '\n';
                if (errors.tellp() > 0) {
                    errors << "; ";
                }
                errors << provider << ": " << ex.what();
            }
        }
        throw std::runtime_error(
            "could not create ONNX Runtime session with any provider: " +
            errors.str()
        );
    }

    void configure_provider(
        const std::string &provider,
        OrtSessionOptions *session_options
    )
    {
        if (provider == "cpu") {
            return;
        }
        if (provider == "cuda") {
            append_cuda_provider(session_options);
            return;
        }
        if (provider == "tensorrt") {
            append_tensorrt_provider(session_options);
            return;
        }
        if (provider == "coreml") {
            append_coreml_provider(session_options);
            return;
        }
        throw std::runtime_error("unsupported ONNX Runtime provider");
    }

    void append_cuda_provider(OrtSessionOptions *session_options)
    {
        OrtCUDAProviderOptionsV2 *options = nullptr;
        const char *keys[] = {"device_id"};
        const char *values[] = {"0"};

        ort_check(api_, api_->CreateCUDAProviderOptions(&options),
                  "CreateCUDAProviderOptions");
        try {
            ort_check(api_, api_->UpdateCUDAProviderOptions(
                options,
                keys,
                values,
                1
            ), "UpdateCUDAProviderOptions");
            ort_check(api_, api_->SessionOptionsAppendExecutionProvider_CUDA_V2(
                session_options,
                options
            ), "SessionOptionsAppendExecutionProvider_CUDA_V2");
        } catch (...) {
            api_->ReleaseCUDAProviderOptions(options);
            throw;
        }
        api_->ReleaseCUDAProviderOptions(options);
    }

    void append_tensorrt_provider(OrtSessionOptions *session_options)
    {
        OrtTensorRTProviderOptionsV2 *options = nullptr;
        const char *keys[] = {"device_id", "trt_fp16_enable"};
        const char *values[] = {
            "0",
            model_.runtime_precision == "fp16" ? "1" : "0"
        };

        ort_check(api_, api_->CreateTensorRTProviderOptions(&options),
                  "CreateTensorRTProviderOptions");
        try {
            ort_check(api_, api_->UpdateTensorRTProviderOptions(
                options,
                keys,
                values,
                2
            ), "UpdateTensorRTProviderOptions");
            ort_check(api_,
                api_->SessionOptionsAppendExecutionProvider_TensorRT_V2(
                    session_options,
                    options
                ),
                "SessionOptionsAppendExecutionProvider_TensorRT_V2");
        } catch (...) {
            api_->ReleaseTensorRTProviderOptions(options);
            throw;
        }
        api_->ReleaseTensorRTProviderOptions(options);
    }

    void append_coreml_provider(OrtSessionOptions *session_options)
    {
#ifdef __APPLE__
        if (std::getenv("OS_ACTIVITY_MODE") == nullptr) {
            setenv("OS_ACTIVITY_MODE", "disable", 0);
        }
#endif
        const char *keys[] = {
            "MLComputeUnits",
            "RequireStaticInputShapes",
            "AllowLowPrecisionAccumulationOnGPU"
        };
        const char *values[] = {
            "CPUAndGPU",
            "1",
            model_.runtime_precision == "fp16" ? "1" : "0"
        };
        ort_check(api_, api_->SessionOptionsAppendExecutionProvider(
            session_options,
            "CoreML",
            keys,
            values,
            3
        ), "SessionOptionsAppendExecutionProvider(CoreML)");
    }

    std::vector<CompiledResult> encode_documents(
        const std::vector<std::string> &texts,
        int max_batch_size
    )
    {
        std::vector<RobertaTokenizer::Windows> tokenized;
        std::vector<WindowRef> refs;
        int semantic_dims = model_.total_dims - model_.lexical_dims;

        tokenized.reserve(texts.size());
        for (const std::string &text : texts) {
            tokenized.push_back(tokenizer_.tokenize_windows(
                text,
                model_.max_length,
                model_.window_stride,
                model_.max_windows
            ));
            for (const auto &window : tokenized.back().windows) {
                refs.push_back({tokenized.size() - 1, &window});
            }
        }
        if (refs.empty()) {
            throw std::runtime_error("tokenizer produced no windows");
        }

        std::vector<float> max_weights(
            texts.size() * static_cast<size_t>(semantic_dims),
            0.0f
        );
        const char *input_names[] = {
            model_.input_ids_name.c_str(),
            model_.attention_mask_name.c_str()
        };
        const char *output_names[] = {
            model_.semantic_ids_name.c_str(),
            model_.semantic_weights_name.c_str()
        };

        for (size_t offset = 0; offset < refs.size();) {
            size_t chunk_count = std::min(
                refs.size() - offset,
                static_cast<size_t>(max_batch_size)
            );
            run_window_chunk(
                refs,
                offset,
                chunk_count,
                semantic_dims,
                input_names,
                output_names,
                max_weights
            );
            offset += chunk_count;
        }

        std::vector<CompiledResult> compiled;
        compiled.reserve(texts.size());
        for (size_t row = 0; row < texts.size(); row++) {
            compiled.push_back(compile_document_row(
                tokenized[row],
                semantic_dims,
                max_weights.data() + row * static_cast<size_t>(semantic_dims)
            ));
        }
        return compiled;
    }

    void run_window_chunk(
        const std::vector<WindowRef> &refs,
        size_t offset,
        size_t chunk_count,
        int semantic_dims,
        const char *const input_names[2],
        const char *const output_names[2],
        std::vector<float> &max_weights
    )
    {
        int sequence_length = 0;
        for (size_t i = 0; i < chunk_count; i++) {
            sequence_length = std::max(
                sequence_length,
                refs[offset + i].input->token_count
            );
        }
        std::vector<int64_t> input_ids(
            chunk_count * static_cast<size_t>(sequence_length),
            0
        );
        std::vector<int64_t> attention_mask(input_ids.size(), 0);
        for (size_t i = 0; i < chunk_count; i++) {
            const auto *input = refs[offset + i].input;
            size_t row_offset = i * static_cast<size_t>(sequence_length);
            std::copy(input->input_ids.begin(), input->input_ids.end(),
                      input_ids.begin() + row_offset);
            std::copy(input->attention_mask.begin(),
                      input->attention_mask.end(),
                      attention_mask.begin() + row_offset);
        }

        OrtMemoryInfo *memory_info = nullptr;
        OrtValue *inputs[2] = {nullptr, nullptr};
        OrtValue *outputs[2] = {nullptr, nullptr};
        OrtTensorTypeAndShapeInfo *atom_shape = nullptr;
        OrtTensorTypeAndShapeInfo *weight_shape = nullptr;
        auto release_all = [&]() {
            if (atom_shape != nullptr) {
                api_->ReleaseTensorTypeAndShapeInfo(atom_shape);
            }
            if (weight_shape != nullptr) {
                api_->ReleaseTensorTypeAndShapeInfo(weight_shape);
            }
            if (outputs[0] != nullptr) {
                api_->ReleaseValue(outputs[0]);
            }
            if (outputs[1] != nullptr) {
                api_->ReleaseValue(outputs[1]);
            }
            if (inputs[0] != nullptr) {
                api_->ReleaseValue(inputs[0]);
            }
            if (inputs[1] != nullptr) {
                api_->ReleaseValue(inputs[1]);
            }
            if (memory_info != nullptr) {
                api_->ReleaseMemoryInfo(memory_info);
            }
        };

        try {
            int64_t shape[] = {
                static_cast<int64_t>(chunk_count),
                static_cast<int64_t>(sequence_length)
            };
            ort_check(api_, api_->CreateCpuMemoryInfo(
                OrtArenaAllocator,
                OrtMemTypeDefault,
                &memory_info
            ), "CreateCpuMemoryInfo");
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                memory_info,
                input_ids.data(),
                input_ids.size() * sizeof(int64_t),
                shape,
                2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                &inputs[0]
            ), "CreateTensor(input_ids)");
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                memory_info,
                attention_mask.data(),
                attention_mask.size() * sizeof(int64_t),
                shape,
                2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                &inputs[1]
            ), "CreateTensor(attention_mask)");
            const OrtValue *input_values[] = {inputs[0], inputs[1]};
            ort_check(api_, api_->Run(
                session_,
                nullptr,
                input_names,
                input_values,
                2,
                output_names,
                2,
                outputs
            ), "Run(document windows)");

            validate_and_accumulate_outputs(
                refs,
                offset,
                chunk_count,
                semantic_dims,
                outputs,
                &atom_shape,
                &weight_shape,
                max_weights
            );
        } catch (...) {
            release_all();
            throw;
        }
        release_all();
    }

    void validate_and_accumulate_outputs(
        const std::vector<WindowRef> &refs,
        size_t offset,
        size_t chunk_count,
        int semantic_dims,
        OrtValue *outputs[2],
        OrtTensorTypeAndShapeInfo **atom_shape_out,
        OrtTensorTypeAndShapeInfo **weight_shape_out,
        std::vector<float> &max_weights
    )
    {
        OrtTensorTypeAndShapeInfo *atom_shape = nullptr;
        OrtTensorTypeAndShapeInfo *weight_shape = nullptr;
        size_t atom_dims_count = 0;
        size_t weight_dims_count = 0;
        int64_t atom_dims[2] = {0, 0};
        int64_t weight_dims[2] = {0, 0};
        size_t atom_count = 0;
        size_t weight_count = 0;
        ONNXTensorElementDataType atom_type;
        ONNXTensorElementDataType weight_type;
        int64_t *atom_data = nullptr;
        float *weight_data = nullptr;

        ort_check(api_, api_->GetTensorTypeAndShape(outputs[0], &atom_shape),
                  "GetTensorTypeAndShape(semantic_ids)");
        *atom_shape_out = atom_shape;
        ort_check(api_, api_->GetTensorTypeAndShape(outputs[1], &weight_shape),
                  "GetTensorTypeAndShape(semantic_weights)");
        *weight_shape_out = weight_shape;
        ort_check(api_, api_->GetDimensionsCount(atom_shape, &atom_dims_count),
                  "GetDimensionsCount(semantic_ids)");
        ort_check(api_, api_->GetDimensionsCount(weight_shape,
                  &weight_dims_count),
                  "GetDimensionsCount(semantic_weights)");
        if (atom_dims_count != 2 || weight_dims_count != 2) {
            throw std::runtime_error("invalid ONNX output rank");
        }
        ort_check(api_, api_->GetDimensions(atom_shape, atom_dims, 2),
                  "GetDimensions(semantic_ids)");
        ort_check(api_, api_->GetDimensions(weight_shape, weight_dims, 2),
                  "GetDimensions(semantic_weights)");
        ort_check(api_, api_->GetTensorShapeElementCount(atom_shape,
                  &atom_count),
                  "GetTensorShapeElementCount(semantic_ids)");
        ort_check(api_, api_->GetTensorShapeElementCount(weight_shape,
                  &weight_count),
                  "GetTensorShapeElementCount(semantic_weights)");
        ort_check(api_, api_->GetTensorElementType(atom_shape, &atom_type),
                  "GetTensorElementType(semantic_ids)");
        ort_check(api_, api_->GetTensorElementType(weight_shape, &weight_type),
                  "GetTensorElementType(semantic_weights)");
        if (atom_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
            weight_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            atom_dims[0] != static_cast<int64_t>(chunk_count) ||
            weight_dims[0] != static_cast<int64_t>(chunk_count) ||
            atom_dims[1] <= 0 ||
            atom_dims[1] > model_.semantic_max_atoms ||
            weight_dims[1] != atom_dims[1] ||
            atom_count != weight_count) {
            throw std::runtime_error("invalid ONNX output shape");
        }
        int semantic_width = static_cast<int>(atom_dims[1]);
        ort_check(api_, api_->GetTensorMutableData(
            outputs[0],
            reinterpret_cast<void **>(&atom_data)
        ), "GetTensorMutableData(semantic_ids)");
        ort_check(api_, api_->GetTensorMutableData(
            outputs[1],
            reinterpret_cast<void **>(&weight_data)
        ), "GetTensorMutableData(semantic_weights)");

        for (size_t i = 0; i < chunk_count; i++) {
            size_t row = refs[offset + i].row;
            size_t semantic_offset = i * static_cast<size_t>(semantic_width);
            size_t dense_offset = row * static_cast<size_t>(semantic_dims);

            for (int j = 0; j < semantic_width; j++) {
                int64_t atom_id = atom_data[semantic_offset + j];
                float weight = weight_data[semantic_offset + j];
                if (atom_id < 0 || atom_id >= semantic_dims) {
                    throw std::runtime_error("semantic atom id out of range");
                }
                if (!std::isfinite(static_cast<double>(weight)) ||
                    weight < 0.0f) {
                    throw std::runtime_error("invalid semantic weight");
                }
                if (weight == 0.0f) {
                    continue;
                }
                float &current = max_weights[
                    dense_offset + static_cast<size_t>(atom_id)
                ];
                if (weight > current) {
                    current = weight;
                }
            }
        }
    }

    CompiledResult compile_document_row(
        const RobertaTokenizer::Windows &tokenized,
        int semantic_dims,
        const float *weights
    ) const
    {
        std::vector<std::pair<int, float>> active;
        CompiledResult result;

        for (int atom_id = 0; atom_id < semantic_dims; atom_id++) {
            float weight = weights[atom_id];
            if (weight == 0.0f) {
                continue;
            }
            if (!std::isfinite(static_cast<double>(weight)) ||
                weight < 0.0f) {
                throw std::runtime_error("invalid document semantic weight");
            }
            active.emplace_back(atom_id, weight);
        }
        if (active.empty()) {
            throw std::runtime_error(
                "document semantic encoder produced no atoms"
            );
        }
        std::sort(active.begin(), active.end(), [](const auto &left,
                                                   const auto &right) {
            if (left.second != right.second) {
                return left.second > right.second;
            }
            return left.first < right.first;
        });
        size_t output_count = std::min(
            active.size(),
            static_cast<size_t>(
                std::min(model_.semantic_max_atoms, model_.max_output_atoms)
            )
        );
        active.resize(output_count);
        std::sort(active.begin(), active.end(), [](const auto &left,
                                                   const auto &right) {
            return left.first < right.first;
        });
        result.atom_ids.reserve(active.size());
        result.atom_weights.reserve(active.size());
        for (const auto &entry : active) {
            result.atom_ids.push_back(model_.lexical_dims + entry.first);
            result.atom_weights.push_back(entry.second);
            result.semantic_proxy += static_cast<double>(entry.second);
        }
        result.token_count = tokenized.full_token_count;
        result.window_count = static_cast<int>(tokenized.windows.size());
        result.window_stride = tokenized.window_stride;
        return result;
    }

    static json_t *compiled_result_json(const CompiledResult &result)
    {
        json_t *root = json_object();
        check_json_alloc(root);
        json_object_set_new(root, "compiler_role", json_string("document"));
        json_t *atoms = json_array();
        json_t *weights = json_array();
        check_json_alloc(atoms);
        check_json_alloc(weights);
        for (int32_t atom_id : result.atom_ids) {
            json_array_append_new(atoms, json_integer(atom_id));
        }
        for (float weight : result.atom_weights) {
            json_array_append_new(weights, json_real(weight));
        }
        json_object_set_new(root, "atoms", atoms);
        json_object_set_new(root, "weights", weights);
        json_object_set_new(root, "token_count", json_integer(
            result.token_count
        ));
        json_object_set_new(root, "window_count", json_integer(
            result.window_count
        ));
        json_object_set_new(root, "window_stride", json_integer(
            result.window_stride
        ));
        json_object_set_new(
            root,
            "aggregation",
            json_string("dimension_max_top_k")
        );
        json_object_set_new(root, "truncated", json_false());
        json_t *compiler = json_object();
        check_json_alloc(compiler);
        json_object_set_new(compiler, "lexical_atoms", json_integer(0));
        json_object_set_new(compiler, "semantic_atoms", json_integer(
            static_cast<json_int_t>(result.atom_ids.size())
        ));
        json_object_set_new(compiler, "lexical_proxy", json_real(0.0));
        json_object_set_new(compiler, "semantic_proxy", json_real(
            result.semantic_proxy
        ));
        json_object_set_new(compiler, "query_scale", json_real(1.0));
        json_object_set_new(root, "compiler", compiler);
        json_object_set_new(root, "atom_count", json_integer(
            static_cast<json_int_t>(result.atom_ids.size())
        ));
        return root;
    }

    ModelInfo &model_;
    const OrtApi *api_;
    RobertaTokenizer tokenizer_;
    OrtEnv *env_ = nullptr;
    OrtSessionOptions *session_options_ = nullptr;
    OrtSession *session_ = nullptr;
    double session_load_ms_ = 0.0;
};

struct WorkItem {
    std::string mode;
    std::vector<std::string> texts;
    std::string expected_checkout_signature;
    std::string expected_runtime_precision;
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::atomic_bool canceled{false};
    int status = 200;
    std::string body;
};

class RuntimeBatcher {
public:
    RuntimeBatcher(
        Options options,
        ModelInfo model
    )
        : options_(std::move(options)), model_(std::move(model))
    {
        worker_count_ = std::clamp(options_.worker_count, 1, MAX_WORKER_COUNT);
        max_queue_depth_ = options_.max_queue_depth;
        if (max_queue_depth_ <= 0) {
            max_queue_depth_ = std::clamp(worker_count_ * 6, 8, 256);
        }
        max_queue_depth_ = std::clamp(max_queue_depth_, 1, MAX_QUEUE_DEPTH);
        max_batch_size_ = std::clamp(
            options_.max_batch_size,
            1,
            MAX_BATCH_SIZE
        );
        engine_ = std::make_shared<DirectRuntimeEngine>(model_, options_);

        for (int i = 0; i < worker_count_; i++) {
            workers_.emplace_back(&RuntimeBatcher::worker_main, this);
        }
    }

    ~RuntimeBatcher()
    {
        {
            std::lock_guard<std::mutex> guard(queue_mutex_);
            stop_ = true;
        }
        queue_cv_.notify_all();
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    RuntimeBatcher(const RuntimeBatcher &) = delete;
    RuntimeBatcher &operator=(const RuntimeBatcher &) = delete;

    const ModelInfo &model() const
    {
        return model_;
    }

    int queue_depth() const
    {
        std::lock_guard<std::mutex> guard(queue_mutex_);
        return static_cast<int>(queue_.size());
    }

    int queue_capacity() const
    {
        return max_queue_depth_;
    }

    int max_batch_size() const
    {
        return max_batch_size_;
    }

    int worker_count() const
    {
        return worker_count_;
    }

    std::string active_provider() const
    {
        return model_.active_provider;
    }

    std::pair<int, std::string> submit(
        const std::string &mode,
        std::vector<std::string> texts,
        const std::string &expected_checkout_signature,
        const std::string &expected_runtime_precision
    )
    {
        validate(
            mode,
            texts,
            expected_checkout_signature,
            expected_runtime_precision
        );

        auto item = std::make_shared<WorkItem>();
        item->mode = mode;
        item->texts = std::move(texts);
        item->expected_checkout_signature = expected_checkout_signature;
        item->expected_runtime_precision = normalize_runtime_precision(
            expected_runtime_precision
        );

        {
            std::lock_guard<std::mutex> guard(queue_mutex_);
            if (queue_.size() >= static_cast<size_t>(max_queue_depth_)) {
                throw HttpError(
                    503,
                    "runtime queue full",
                    "pending request count reached " +
                        std::to_string(max_queue_depth_)
                );
            }
            queue_.push_back(item);
        }
        queue_cv_.notify_one();

        std::unique_lock<std::mutex> lock(item->mutex);
        bool completed = item->cv.wait_for(
            lock,
            std::chrono::seconds(options_.request_timeout_s),
            [&item] { return item->done; }
        );
        if (!completed) {
            item->canceled.store(true);
            throw HttpError(504, "runtime request timed out");
        }
        return {item->status, item->body};
    }

private:
    void validate(
        const std::string &mode,
        const std::vector<std::string> &texts,
        const std::string &expected_checkout_signature,
        const std::string &expected_runtime_precision
    ) const
    {
        if (mode != "document" && mode != "query") {
            throw HttpError(400, "unsupported encode mode", mode);
        }
        if (expected_checkout_signature != model_.checkout_signature) {
            throw HttpError(
                409,
                "model checkout mismatch",
                "expected " + expected_checkout_signature +
                    ", service has " + model_.checkout_signature
            );
        }
        std::string normalized_precision;
        try {
            normalized_precision =
                normalize_runtime_precision(expected_runtime_precision);
        } catch (const std::exception &error) {
            throw HttpError(400, "invalid runtime precision", error.what());
        }
        if (normalized_precision != model_.runtime_precision) {
            throw HttpError(
                409,
                "runtime precision mismatch",
                "expected " + expected_runtime_precision +
                    ", service has " + model_.runtime_precision
            );
        }
        if (texts.empty() ||
            texts.size() > static_cast<size_t>(max_batch_size_)) {
            throw HttpError(
                400,
                "invalid text batch size",
                "count must be between 1 and " +
                    std::to_string(max_batch_size_)
            );
        }
        for (size_t i = 0; i < texts.size(); i++) {
            if (texts[i].empty()) {
                throw HttpError(
                    400,
                    "text entries must be non-empty strings",
                    "offset=" + std::to_string(i)
                );
            }
            if (texts[i].size() + 1 > RUNTIME_TRANSPORT_MAX_BYTES) {
                throw HttpError(
                    400,
                    "text entry exceeds runtime transport limit",
                    "offset=" + std::to_string(i)
                );
            }
        }
    }

    std::shared_ptr<WorkItem> pop_first()
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        queue_cv_.wait(lock, [this] {
            return stop_ || !queue_.empty();
        });
        if (stop_ && queue_.empty()) {
            return nullptr;
        }
        auto item = queue_.front();
        queue_.pop_front();
        return item;
    }

    void worker_main()
    {
        while (!stop_requested) {
            auto first = pop_first();
            if (first == nullptr) {
                return;
            }
            if (first->canceled.load()) {
                continue;
            }
            std::vector<std::shared_ptr<WorkItem>> items{first};
            size_t total = first->texts.size();
            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(options_.max_delay_ms);

            while (total < static_cast<size_t>(max_batch_size_)) {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (queue_.empty()) {
                    if (queue_cv_.wait_until(lock, deadline) ==
                        std::cv_status::timeout) {
                        break;
                    }
                }
                if (queue_.empty()) {
                    continue;
                }
                auto candidate = queue_.front();
                if (candidate->canceled.load()) {
                    queue_.pop_front();
                    continue;
                }
                bool compatible =
                    candidate->mode == first->mode &&
                    candidate->expected_checkout_signature ==
                        first->expected_checkout_signature &&
                    candidate->expected_runtime_precision ==
                        first->expected_runtime_precision &&
                    total + candidate->texts.size() <=
                        static_cast<size_t>(max_batch_size_);
                if (!compatible) {
                    break;
                }
                queue_.pop_front();
                total += candidate->texts.size();
                items.push_back(candidate);
            }
            execute_batch(items);
        }
    }

    void execute_batch(
        const std::vector<std::shared_ptr<WorkItem>> &items
    )
    {
        std::vector<std::string> texts;
        std::vector<std::pair<size_t, size_t>> ranges;

        for (const auto &item : items) {
            size_t start = texts.size();
            texts.insert(texts.end(), item->texts.begin(), item->texts.end());
            ranges.push_back({start, texts.size()});
        }

        try {
            JsonPtr envelope;
            JsonPtr results(json_array());

            execute_transport_chunks(
                items[0]->mode,
                texts,
                envelope,
                results.get()
            );
            for (size_t i = 0; i < items.size(); i++) {
                set_item_result(
                    items[i],
                    200,
                    response_envelope(
                        items[0]->mode,
                        envelope.get(),
                        results.get(),
                        ranges[i].first,
                        ranges[i].second
                    )
                );
            }
        } catch (const HttpError &error) {
            for (const auto &item : items) {
                set_item_result(
                    item,
                    error.status,
                    json_error_body(error.what(), error.detail)
                );
            }
        } catch (const std::exception &error) {
            for (const auto &item : items) {
                set_item_result(
                    item,
                    500,
                    json_error_body("runtime request failed", error.what())
                );
            }
        }
    }

    void execute_transport_chunks(
        const std::string &mode,
        const std::vector<std::string> &texts,
        JsonPtr &envelope_out,
        json_t *results_out
    )
    {
        size_t chunk_start = 0;

        if (!json_is_array(results_out)) {
            throw HttpError(500, "runtime result accumulator is invalid");
        }
        while (chunk_start < texts.size()) {
            size_t chunk_end = chunk_start;
            size_t chunk_bytes = 0;
            std::vector<std::string> chunk_texts;

            while (chunk_end < texts.size() &&
                   chunk_texts.size() < static_cast<size_t>(max_batch_size_)) {
                size_t text_bytes = texts[chunk_end].size() + 1;

                if (text_bytes > RUNTIME_TRANSPORT_MAX_BYTES) {
                    throw HttpError(
                        400,
                        "text entry exceeds runtime transport limit",
                        "offset=" + std::to_string(chunk_end)
                    );
                }
                if (!chunk_texts.empty() &&
                    chunk_bytes + text_bytes > RUNTIME_TRANSPORT_MAX_BYTES) {
                    break;
                }
                chunk_bytes += text_bytes;
                chunk_texts.push_back(texts[chunk_end]);
                chunk_end++;
            }
            if (chunk_texts.empty()) {
                throw HttpError(500, "runtime transport chunk is empty");
            }
            std::string envelope_json = engine_->encode(
                mode,
                chunk_texts,
                max_batch_size_
            );
            JsonPtr envelope(parse_json(envelope_json));

            validate_runtime_envelope(envelope.get(), chunk_texts.size());
            json_t *chunk_results = json_object_get(
                envelope.get(),
                "results"
            );
            for (size_t i = 0; i < json_array_size(chunk_results); i++) {
                json_array_append(results_out, json_array_get(chunk_results, i));
            }
            if (envelope_out.get() == nullptr) {
                envelope_out = std::move(envelope);
            }
            chunk_start = chunk_end;
        }
        if (json_array_size(results_out) != texts.size()) {
            throw HttpError(502, "runtime returned an invalid result count");
        }
    }

    void validate_runtime_envelope(json_t *envelope, size_t expected_count)
    {
        json_t *checkout = json_object_get(envelope, "checkout_signature");
        const char *checkout_value = json_string_value(checkout);
        if (checkout_value == nullptr ||
            model_.checkout_signature != checkout_value) {
            throw HttpError(
                502,
                "runtime returned the wrong checkout",
                model_.checkout_signature
            );
        }
        json_t *precision = json_object_get(envelope, "runtime_precision");
        const char *precision_value = json_string_value(precision);
        if (precision_value == nullptr ||
            model_.runtime_precision != precision_value) {
            throw HttpError(
                502,
                "runtime returned the wrong precision",
                model_.runtime_precision
            );
        }

        json_t *results = json_object_get(envelope, "results");
        if (!json_is_array(results) ||
            json_array_size(results) != expected_count) {
            throw HttpError(502, "runtime returned an invalid result count");
        }
    }

    json_t *parse_json(const std::string &body) const
    {
        json_error_t error;
        json_t *root = json_loadb(body.data(), body.size(), 0, &error);
        if (root == nullptr || !json_is_object(root)) {
            json_decref(root);
            throw HttpError(502, "runtime returned invalid JSON", error.text);
        }
        return root;
    }

    std::string response_envelope(
        const std::string &mode,
        json_t *runtime_envelope,
        json_t *batch_results,
        size_t start,
        size_t end
    ) const
    {
        JsonPtr root(json_object());
        check_json_alloc(root.get());
        json_set_string(root.get(), "runtime_backend", "direct");
        json_set_string(root.get(), "model_id", model_.model_id);
        json_set_string(root.get(), "model_path", model_.model_path);
        json_set_string(
            root.get(),
            "checkout_signature",
            model_.checkout_signature
        );
        json_set_string(root.get(), "runtime_abi", model_.runtime_abi);
        json_set_string(
            root.get(),
            "runtime_precision",
            model_.runtime_precision
        );
        json_object_set_new(
            root.get(),
            "runtime_output",
            json_deep_copy(model_.runtime_output.get())
        );
        json_set_string(
            root.get(),
            "provider",
            runtime_provider_string_default(
                runtime_envelope,
                "requested",
                json_string_default(
                    runtime_envelope,
                    "provider",
                    model_.provider
                )
            )
        );
        json_set_string(
            root.get(),
            "active_provider",
            runtime_provider_string_default(
                runtime_envelope,
                "active",
                json_string_default(
                    runtime_envelope,
                    "active_provider",
                    model_.active_provider
                )
            )
        );
        json_set_string(root.get(), "mode", mode);
        json_copy_field(root.get(), runtime_envelope, "batch_size");
        json_copy_field(root.get(), runtime_envelope, "sequence_length");
        json_copy_field(root.get(), runtime_envelope, "window_count");
        json_copy_field(root.get(), runtime_envelope, "latency_ms");
        json_copy_field(root.get(), runtime_envelope, "session_cache_hit");
        json_copy_field(root.get(), runtime_envelope, "session_load_ms");

        json_t *results = json_array();
        check_json_alloc(results);
        for (size_t i = start; i < end; i++) {
            json_array_append(results, json_array_get(batch_results, i));
        }
        json_object_set_new(root.get(), "results", results);
        return dump_json(root.get());
    }

    void set_item_result(
        const std::shared_ptr<WorkItem> &item,
        int status,
        const std::string &body
    )
    {
        {
            std::lock_guard<std::mutex> guard(item->mutex);
            item->status = status;
            item->body = body;
            item->done = true;
        }
        item->cv.notify_one();
    }

    Options options_;
    ModelInfo model_;
    std::shared_ptr<DirectRuntimeEngine> engine_;
    int max_batch_size_ = DEFAULT_MAX_BATCH_SIZE;
    int worker_count_ = DEFAULT_WORKER_COUNT;
    int max_queue_depth_ = 8;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<WorkItem>> queue_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

void send_all(int fd, const std::string &data)
{
    const char *ptr = data.data();
    size_t remaining = data.size();

    while (remaining > 0) {
        ssize_t written = send(fd, ptr, remaining, 0);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        if (written == 0) {
            return;
        }
        ptr += written;
        remaining -= static_cast<size_t>(written);
    }
}

void write_response(
    int fd,
    int status,
    const std::string &body,
    const std::string &content_type = JSON_CONTENT_TYPE,
    bool keep_alive = false
)
{
    std::ostringstream response;
    const char *reason = "OK";

    if (status == 400) {
        reason = "Bad Request";
    } else if (status == 404) {
        reason = "Not Found";
    } else if (status == 409) {
        reason = "Conflict";
    } else if (status == 500) {
        reason = "Internal Server Error";
    } else if (status == 502) {
        reason = "Bad Gateway";
    } else if (status == 503) {
        reason = "Service Unavailable";
    } else if (status == 504) {
        reason = "Gateway Timeout";
    }
    response << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
             << "Content-Type: " << content_type << "\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Connection: "
             << (keep_alive ? "keep-alive" : "close")
             << "\r\n\r\n"
             << body;
    send_all(fd, response.str());
}

void write_error(
    int fd,
    int status,
    const std::string &message,
    const std::string &detail = "",
    bool keep_alive = false
)
{
    write_response(
        fd,
        status,
        json_error_body(message, detail),
        JSON_CONTENT_TYPE,
        keep_alive
    );
}

std::string lower_header_name(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });
    return value;
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    bool close_after_response = false;
};

HttpRequest read_request(int fd, size_t max_request_bytes)
{
    std::string raw;
    char buffer[4096];
    size_t header_end = std::string::npos;

    while (header_end == std::string::npos) {
        ssize_t got = recv(fd, buffer, sizeof(buffer), 0);
        if (got <= 0) {
            throw HttpError(400, "empty request");
        }
        raw.append(buffer, static_cast<size_t>(got));
        if (raw.size() > 64 * 1024) {
            throw HttpError(400, "request header is too large");
        }
        header_end = raw.find("\r\n\r\n");
    }

    std::string header = raw.substr(0, header_end);
    std::istringstream header_stream(header);
    std::string request_line;
    HttpRequest request;
    size_t content_length = 0;

    std::getline(header_stream, request_line);
    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }
    {
        std::istringstream line_stream(request_line);
        std::string version;
        line_stream >> request.method >> request.path >> version;
        if (request.method.empty() || request.path.empty()) {
            throw HttpError(400, "invalid request line");
        }
    }

    std::string line;
    while (std::getline(header_stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        size_t separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        std::string name = lower_header_name(line.substr(0, separator));
        std::string value = line.substr(separator + 1);
        while (!value.empty() && value.front() == ' ') {
            value.erase(value.begin());
        }
        if (name == "content-length") {
            try {
                content_length = static_cast<size_t>(std::stoul(value));
            } catch (const std::exception &) {
                throw HttpError(400, "invalid content length");
            }
        } else if (name == "connection") {
            std::string connection = lower_header_name(value);
            if (connection == "close") {
                request.close_after_response = true;
            }
        }
    }
    if (content_length > max_request_bytes) {
        throw HttpError(400, "invalid request body size");
    }

    size_t body_start = header_end + 4;
    while (raw.size() < body_start + content_length) {
        ssize_t got = recv(fd, buffer, sizeof(buffer), 0);
        if (got <= 0) {
            throw HttpError(400, "incomplete request body");
        }
        raw.append(buffer, static_cast<size_t>(got));
    }
    request.body = raw.substr(body_start, content_length);
    return request;
}

std::vector<std::string> read_texts(json_t *payload)
{
    json_t *texts_json = json_object_get(payload, "texts");
    std::vector<std::string> texts;

    if (!json_is_array(texts_json)) {
        throw HttpError(400, "texts must be an array");
    }
    size_t index;
    json_t *value;
    json_array_foreach(texts_json, index, value) {
        const char *text = json_string_value(value);
        if (text == nullptr) {
            throw HttpError(
                400,
                "text entries must be non-empty strings",
                "offset=" + std::to_string(index)
            );
        }
        texts.emplace_back(text);
    }
    return texts;
}

std::string read_string_default(
    json_t *payload,
    const char *key,
    const std::string &default_value
)
{
    json_t *value = json_object_get(payload, key);

    if (value == nullptr) {
        return default_value;
    }
    if (!json_is_string(value)) {
        throw HttpError(400, std::string(key) + " must be a string");
    }
    return json_string_value(value);
}

std::string health_body(const RuntimeBatcher &batcher)
{
    JsonPtr root(json_object());
    check_json_alloc(root.get());
    json_object_set_new(root.get(), "ok", json_true());
    json_set_string(root.get(), "runtime_backend", "direct");
    json_set_string(root.get(), "model_id", batcher.model().model_id);
    json_set_string(
        root.get(),
        "checkout_signature",
        batcher.model().checkout_signature
    );
    json_set_string(
        root.get(),
        "runtime_precision",
        batcher.model().runtime_precision
    );
    json_set_string(root.get(), "active_provider", batcher.active_provider());
    json_object_set_new(root.get(), "queue_depth", json_integer(
        batcher.queue_depth()
    ));
    json_object_set_new(root.get(), "queue_capacity", json_integer(
        batcher.queue_capacity()
    ));
    json_object_set_new(root.get(), "worker_count", json_integer(
        batcher.worker_count()
    ));
    json_object_set_new(root.get(), "max_batch_size", json_integer(
        batcher.max_batch_size()
    ));
    return dump_json(root.get());
}

std::string model_body(const RuntimeBatcher &batcher)
{
    const ModelInfo &model = batcher.model();
    JsonPtr root(json_object());

    check_json_alloc(root.get());
    json_set_string(root.get(), "runtime_backend", "direct");
    json_set_string(root.get(), "model_id", model.model_id);
    json_set_string(root.get(), "model_path", model.model_path);
    json_set_string(root.get(), "checkout_signature", model.checkout_signature);
    json_set_string(root.get(), "runtime_precision", model.runtime_precision);
    json_set_string(root.get(), "runtime_abi", model.runtime_abi);
    json_object_set_new(
        root.get(),
        "runtime_output",
        json_deep_copy(model.runtime_output.get())
    );
    json_set_string(root.get(), "provider", model.provider);
    json_set_string(root.get(), "active_provider", batcher.active_provider());
    json_object_set_new(
        root.get(),
        "max_batch_size",
        json_integer(batcher.max_batch_size())
    );
    json_object_set_new(
        root.get(),
        "worker_count",
        json_integer(batcher.worker_count())
    );
    return dump_json(root.get());
}

struct ActiveConnectionGuard {
    explicit ActiveConnectionGuard(
        std::shared_ptr<std::atomic<int>> active_connections
    ) : active_connections_(std::move(active_connections))
    {
    }

    ~ActiveConnectionGuard()
    {
        active_connections_->fetch_sub(1, std::memory_order_relaxed);
    }

private:
    std::shared_ptr<std::atomic<int>> active_connections_;
};

void handle_connection(
    int fd,
    std::shared_ptr<RuntimeBatcher> batcher,
    size_t max_request_bytes,
    std::shared_ptr<std::atomic<int>> active_connections
)
{
    ActiveConnectionGuard connection_guard(std::move(active_connections));

    for (;;) {
        try {
            HttpRequest request = read_request(fd, max_request_bytes);
            bool keep_alive = !request.close_after_response;

            if (request.method == "GET" && request.path == "/health") {
                write_response(
                    fd,
                    200,
                    health_body(*batcher),
                    JSON_CONTENT_TYPE,
                    keep_alive
                );
            } else if (
                request.method == "GET" &&
                request.path == "/v1/models"
            ) {
                write_response(
                    fd,
                    200,
                    model_body(*batcher),
                    JSON_CONTENT_TYPE,
                    keep_alive
                );
            } else if (
                request.method == "POST" &&
                request.path == "/v1/encode"
            ) {
                JsonPtr payload = parse_json_body(request.body);
                std::string mode = read_string_default(
                    payload.get(),
                    "mode",
                    "document"
                );
                std::string checkout = read_string_default(
                    payload.get(),
                    "checkout_signature",
                    ""
                );
                std::string runtime_precision = read_string_default(
                    payload.get(),
                    "runtime_precision",
                    ""
                );
                auto result = batcher->submit(
                    mode,
                    read_texts(payload.get()),
                    checkout,
                    runtime_precision
                );
                write_response(
                    fd,
                    result.first,
                    result.second,
                    JSON_CONTENT_TYPE,
                    keep_alive
                );
            } else {
                write_error(fd, 404, "not found", "", keep_alive);
            }
            if (!keep_alive) {
                break;
            }
        } catch (const HttpError &error) {
            write_error(fd, error.status, error.what(), error.detail);
            break;
        } catch (const std::exception &error) {
            write_error(fd, 500, "internal server error", error.what());
            break;
        }
    }
    close(fd);
}

bool configure_connection_socket(int fd, int idle_timeout_s)
{
    timeval timeout{};

    timeout.tv_sec = idle_timeout_s;
    return setsockopt(
        fd,
        SOL_SOCKET,
        SO_RCVTIMEO,
        &timeout,
        sizeof(timeout)
    ) == 0 && setsockopt(
        fd,
        SOL_SOCKET,
        SO_SNDTIMEO,
        &timeout,
        sizeof(timeout)
    ) == 0;
}

void handle_signal(int)
{
    stop_requested = 1;
    if (listen_fd >= 0) {
        close(listen_fd);
        listen_fd = -1;
    }
}

int bind_listener(const std::string &host, int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int enabled = 1;
    sockaddr_in address;

    if (fd < 0) {
        throw std::runtime_error("could not create socket");
    }
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
        close(fd);
        throw std::runtime_error("host must be an IPv4 address");
    }
    if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
        std::string error = std::strerror(errno);
        close(fd);
        throw std::runtime_error("could not bind listener: " + error);
    }
    if (listen(fd, LISTEN_BACKLOG) < 0) {
        std::string error = std::strerror(errno);
        close(fd);
        throw std::runtime_error("could not listen: " + error);
    }
    return fd;
}

void print_help(const char *program)
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --model-path PATH\n"
        << "  --checkout-signature HEX\n"
        << "  --runtime-precision fp16|fp32\n"
        << "  --host IPV4\n"
        << "  --port PORT\n"
        << "  --max-batch-size N\n"
        << "  --max-delay-ms N\n"
        << "  --request-timeout-s N\n"
        << "  --worker-count N\n"
        << "  --max-queue-depth N\n"
        << "  --max-connections N\n"
        << "  --connection-idle-timeout-s N\n"
        << "  --max-request-bytes N\n"
        << "  --intra-op-threads N\n";
}

int parse_positive_int(const std::string &value, const char *name)
{
    int parsed = std::stoi(value);

    if (parsed <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return parsed;
}

Options parse_args(int argc, char **argv)
{
    Options options;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto require_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires value");
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--dsn") {
            (void) require_value("--dsn");
            throw std::runtime_error(
                "--dsn is not supported by the direct runtime server"
            );
        } else if (arg == "--model-path") {
            options.model_path = require_value("--model-path");
        } else if (arg == "--checkout-signature") {
            options.checkout_signature = require_value("--checkout-signature");
        } else if (arg == "--runtime-precision") {
            options.runtime_precision = normalize_runtime_precision(
                require_value("--runtime-precision")
            );
        } else if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_positive_int(require_value("--port"), "--port");
        } else if (arg == "--max-batch-size") {
            options.max_batch_size = parse_positive_int(
                require_value("--max-batch-size"),
                "--max-batch-size"
            );
        } else if (arg == "--max-delay-ms") {
            options.max_delay_ms = parse_positive_int(
                require_value("--max-delay-ms"),
                "--max-delay-ms"
            );
        } else if (arg == "--request-timeout-s") {
            options.request_timeout_s = parse_positive_int(
                require_value("--request-timeout-s"),
                "--request-timeout-s"
            );
        } else if (arg == "--worker-count") {
            options.worker_count = parse_positive_int(
                require_value("--worker-count"),
                "--worker-count"
            );
        } else if (arg == "--max-queue-depth") {
            options.max_queue_depth = parse_positive_int(
                require_value("--max-queue-depth"),
                "--max-queue-depth"
            );
        } else if (arg == "--max-connections") {
            options.max_connections = parse_positive_int(
                require_value("--max-connections"),
                "--max-connections"
            );
        } else if (arg == "--connection-idle-timeout-s") {
            options.connection_idle_timeout_s = parse_positive_int(
                require_value("--connection-idle-timeout-s"),
                "--connection-idle-timeout-s"
            );
        } else if (arg == "--max-request-bytes") {
            options.max_request_bytes = static_cast<size_t>(
                parse_positive_int(
                    require_value("--max-request-bytes"),
                    "--max-request-bytes"
                )
            );
        } else if (arg == "--intra-op-threads") {
            options.intra_op_threads = parse_positive_int(
                require_value("--intra-op-threads"),
                "--intra-op-threads"
            );
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        Options options = parse_args(argc, argv);
        ModelInfo model = load_model(options);
        auto batcher = std::make_shared<RuntimeBatcher>(
            options,
            std::move(model)
        );
        auto active_connections = std::make_shared<std::atomic<int>>(0);
        int max_connections = options.max_connections > 0
            ? std::clamp(options.max_connections, 1, MAX_CONNECTIONS)
            : std::clamp(
                batcher->queue_capacity() + batcher->worker_count() * 2,
                64,
                MAX_CONNECTIONS
            );

        signal(SIGINT, handle_signal);
        signal(SIGTERM, handle_signal);
        signal(SIGPIPE, SIG_IGN);
        listen_fd = bind_listener(options.host, options.port);
        std::cerr
            << "ii42-runtime-server listening on "
            << options.host << ':' << options.port
            << " model_id=" << batcher->model().model_id
            << " checkout=" << batcher->model().checkout_signature
            << " precision=" << batcher->model().runtime_precision
            << '\n';

        while (!stop_requested) {
            sockaddr_in client_address;
            socklen_t client_len = sizeof(client_address);
            int client_fd = accept(
                listen_fd,
                reinterpret_cast<sockaddr *>(&client_address),
                &client_len
            );
            if (client_fd < 0) {
                if (errno == EINTR) {
                    continue;
                }
                if (stop_requested) {
                    break;
                }
                throw std::runtime_error(
                    "accept failed: " + std::string(std::strerror(errno))
                );
            }
            int previous_connections = active_connections->fetch_add(
                1,
                std::memory_order_relaxed
            );
            if (previous_connections >= max_connections) {
                active_connections->fetch_sub(1, std::memory_order_relaxed);
                if (configure_connection_socket(
                        client_fd,
                        options.connection_idle_timeout_s)) {
                    write_error(
                        client_fd,
                        503,
                        "connection capacity exhausted"
                    );
                }
                close(client_fd);
                continue;
            }
            if (!configure_connection_socket(
                    client_fd,
                    options.connection_idle_timeout_s)) {
                active_connections->fetch_sub(1, std::memory_order_relaxed);
                close(client_fd);
                continue;
            }
            std::thread(
                handle_connection,
                client_fd,
                batcher,
                options.max_request_bytes,
                active_connections
            ).detach();
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ii42-runtime-server: " << error.what() << '\n';
        return 1;
    }
}
