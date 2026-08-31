#include "storage.h"
#include "allocator.h"
#include <stdexcept>

Storage::Storage(size_t total_elements, Device d, DType type) 
    : device(d), dtype(type) {
        size_t element_size; 
        switch(type) {
            case DType::Float32:  element_size = 4; break;
            case DType::Float16:  element_size = 2; break;
            case DType::BFloat16: element_size = 2; break;
            case DType::Int8:     element_size = 1; break;
            default: throw std::runtime_error("Unknown DType!");
        }
        total_bytes = total_elements * element_size;
        if(device == Device::CPU) 
            cpu_data = get_memory(device, total_bytes);
}

Storage::~Storage() {
    if(cpu_data) free_memory(device, cpu_data, total_bytes);
}
