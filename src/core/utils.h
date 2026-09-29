#pragma once
#include <vector>
#include <cstdint>

std::vector<int64_t> make_strides(const std::vector<int64_t>& shape);
std::vector<int64_t> unravel(int64_t flat, const std::vector<int64_t>& shape);
