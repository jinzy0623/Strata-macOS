#include "strata/platform/direct_file.hpp"
#include "strata/platform/memory.hpp"
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/resource.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
int main() {
    using namespace strata::platform;
    char path[] = "/tmp/strata-io-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    char data[4096];
    std::memset(data, 0x5a, sizeof(data));
    CHECK(write(fd, data, sizeof(data)) == sizeof(data));
    CHECK(write(fd, data, 17) == 17);
    ::close(fd);
    DirectFile file;
    std::string error;
    CHECK(file.open(path, error));
    unlink(path);
    CHECK(file.size() == 4113);
    void* buffer = DirectFile::alloc_aligned(4096);
    CHECK(buffer);
    CHECK(!file.submit(1, buffer, 4096, 1, error));
    CHECK(file.submit(0, buffer, 4096, 42, error));
    Completion done;
    CHECK(file.wait(&done, 1, 0) == 1 && done.ok && done.bytes == 4096 && done.tag == 42);
    CHECK(std::memcmp(data, buffer, 4096) == 0);
    CHECK(file.submit(4096, buffer, 4096, 43, error));
    CHECK(file.wait(&done, 1, 0) == 1 && done.ok && done.bytes == 17);
    CHECK(!lock_resident(nullptr, 0).ok);
    auto locked = lock_resident(buffer, 4096);
    CHECK(locked.ok ? locked.locked_bytes == 4096 : locked.locked_bytes == 0 && !locked.note.empty());
    if (locked.ok) unlock_resident(buffer, 4096);
    // Exercise the failure contract without trying to allocate gigabytes.
    struct rlimit limit;
    CHECK(getrlimit(RLIMIT_MEMLOCK, &limit) == 0);
    limit.rlim_cur = 0;
    CHECK(setrlimit(RLIMIT_MEMLOCK, &limit) == 0);
    auto refused = lock_resident(buffer, 4096);
    CHECK(!refused.ok && refused.locked_bytes == 0 && !refused.note.empty());
    DirectFile::free_aligned(buffer);
    file.close();
    CHECK(!file.is_open());
    CHECK(!file.open("/nonexistent/strata-test", error));
    std::puts("macOS file reads, EOF, alignment and optional locking: OK");
}
