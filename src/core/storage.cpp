#include "storage.h"
#include "allocator.h"
#include <stdexcept>

Storage::Storage(size_t total_elements, Device d, DType type) 
    : device(d), dtype(type) {
    total_bytes = total_elements * dtype_size(type);
    data = get_memory(device, total_bytes);
}

Storage::~Storage() {
    if(data) free_memory(device, data, total_bytes);
}
