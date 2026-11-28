#pragma once
// Byte-pair-encoding tokenizers: trained here, or read from a Hugging Face tokenizer.json (GPT-2, Llama, Mistral,
// Qwen2 and others of their kind).
//
//   auto tok = Tokenizer::from_file("Llama-3.2-1B/tokenizer.json");
//   auto ids = tok.encode("The capital of France is");
//   std::cout << tok.decode(ids);
//
//   auto mine = Tokenizer::train(text, 4096, {"<|end|>"});
//   mine.save("tokenizer.json");
//
// Text is split into words before merging, by rules that ask whether a character is a letter, a number or a
// space. That is exact for ASCII and for the Latin, Greek, Cyrillic, Hebrew, Arabic, Korean, Japanese and Chinese
// letters, and an approximation elsewhere, where a word may be split differently than by the original. Decoding is
// always exact. Unicode normalization (NFC) is not applied.

#include "json.h"
#include "tensor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokenizer_detail {

// One character of UTF-8 text at i; a malformed byte counts as a character of its own
inline uint32_t next_char(const std::string& s, size_t i, size_t* length) {
    auto byte = [&](size_t k) { return (uint32_t)(unsigned char)s[k]; };
    uint32_t b = byte(i);
    size_t n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
    *length = 1;
    if (n <= 1 || i + n > s.size()) return b;
    uint32_t cp = b & (0xFFu >> (n + 1));
    for (size_t k = 1; k < n; ++k) {
        if ((byte(i + k) >> 6) != 2) return b;
        cp = (cp << 6) | (byte(i + k) & 0x3F);
    }
    *length = n;
    return cp;
}

inline std::string to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

inline bool is_space(uint32_t c) {
    return (c >= 9 && c <= 13) || c == ' ' || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
inline bool in_ranges(uint32_t c, const std::pair<uint32_t, uint32_t>* begin, const std::pair<uint32_t, uint32_t>* end) {
    auto it = std::upper_bound(begin, end, c, [](uint32_t v, const std::pair<uint32_t, uint32_t>& r) { return v < r.first; });
    return it != begin && c <= (it - 1)->second;
}
inline bool is_number(uint32_t c) {
    if (c < 0x80) return c >= '0' && c <= '9';
    static const std::pair<uint32_t, uint32_t> r[] = {
        {0xB2, 0xB3}, {0xB9, 0xB9}, {0xBC, 0xBE}, {0x660, 0x669}, {0x6F0, 0x6F9}, {0x7C0, 0x7C9}, {0x966, 0x96F}, {0x9E6, 0x9EF},
        {0xA66, 0xA6F}, {0xAE6, 0xAEF}, {0xB66, 0xB6F}, {0xBE6, 0xBF2}, {0xC66, 0xC6F}, {0xCE6, 0xCEF}, {0xD66, 0xD75}, {0xE50, 0xE59},
        {0xED0, 0xED9}, {0xF20, 0xF33}, {0x1040, 0x1049}, {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x2070, 0x2070}, {0x2074, 0x2079}, {0x2080, 0x2089},
        {0x2150, 0x2189}, {0x2460, 0x249B}, {0x24EA, 0x24FF}, {0x2776, 0x2793}, {0x3007, 0x3007}, {0x3021, 0x3029}, {0x3038, 0x303A}, {0x3192, 0x3195},
        {0x3220, 0x3229}, {0x3248, 0x324F}, {0x3251, 0x325F}, {0x3280, 0x3289}, {0x32B1, 0x32BF}, {0xFF10, 0xFF19}, {0x1D7CE, 0x1D7FF}};
    return in_ranges(c, std::begin(r), std::end(r));
}
inline bool is_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    static const std::pair<uint32_t, uint32_t> r[] = {
        {0xAA, 0xAA}, {0xB5, 0xB5}, {0xBA, 0xBA}, {0xC0, 0xD6}, {0xD8, 0xF6}, {0xF8, 0x2C1}, {0x2C6, 0x2D1}, {0x2E0, 0x2E4}, {0x2EC, 0x2EC}, {0x2EE, 0x2EE},
        {0x370, 0x374}, {0x376, 0x377}, {0x37A, 0x37D}, {0x37F, 0x37F}, {0x386, 0x386}, {0x388, 0x38A}, {0x38C, 0x38C}, {0x38E, 0x3A1}, {0x3A3, 0x3F5}, {0x3F7, 0x481},
        {0x48A, 0x52F}, {0x531, 0x556}, {0x559, 0x559}, {0x560, 0x588}, {0x5D0, 0x5EA}, {0x5EF, 0x5F2}, {0x620, 0x64A}, {0x66E, 0x66F}, {0x671, 0x6D3}, {0x6D5, 0x6D5},
        {0x6E5, 0x6E6}, {0x6EE, 0x6EF}, {0x6FA, 0x6FC}, {0x6FF, 0x6FF}, {0x710, 0x710}, {0x712, 0x72F}, {0x74D, 0x7A5}, {0x7B1, 0x7B1}, {0x7CA, 0x7EA}, {0x800, 0x815},
        {0x840, 0x858}, {0x8A0, 0x8C9}, {0x904, 0x939}, {0x93D, 0x93D}, {0x950, 0x950}, {0x958, 0x961}, {0x971, 0x980}, {0x985, 0x9B9}, {0x9BD, 0x9BD}, {0x9DC, 0x9E1},
        {0x9F0, 0x9F1}, {0xA05, 0xA39}, {0xA59, 0xA5E}, {0xA72, 0xA74}, {0xA85, 0xAB9}, {0xABD, 0xABD}, {0xAD0, 0xAD0}, {0xAE0, 0xAE1}, {0xB05, 0xB39}, {0xB3D, 0xB3D},
        {0xB5C, 0xB61}, {0xB83, 0xBB9}, {0xBD0, 0xBD0}, {0xC05, 0xC39}, {0xC3D, 0xC3D}, {0xC58, 0xC61}, {0xC85, 0xCB9}, {0xCBD, 0xCBD}, {0xCDD, 0xCE1}, {0xD05, 0xD3A},
        {0xD3D, 0xD3D}, {0xD4E, 0xD4E}, {0xD54, 0xD56}, {0xD5F, 0xD61}, {0xD7A, 0xD7F}, {0xD85, 0xDC6}, {0xE01, 0xE30}, {0xE32, 0xE33}, {0xE40, 0xE46}, {0xE81, 0xEB0},
        {0xEB2, 0xEB3}, {0xEBD, 0xEBD}, {0xEC0, 0xEC6}, {0xEDC, 0xEDF}, {0xF00, 0xF00}, {0xF40, 0xF6C}, {0xF88, 0xF8C}, {0x1000, 0x102A}, {0x103F, 0x103F}, {0x1050, 0x1055},
        {0x10A0, 0x10FA}, {0x10FC, 0x1248}, {0x124A, 0x135A}, {0x1380, 0x138F}, {0x13A0, 0x13FD}, {0x1401, 0x166C}, {0x166F, 0x167F}, {0x1681, 0x169A}, {0x16A0, 0x16EA},
        {0x16F1, 0x16F8}, {0x1700, 0x1711}, {0x1780, 0x17B3}, {0x17D7, 0x17D7}, {0x1820, 0x1878}, {0x1880, 0x18AA}, {0x1C80, 0x1CBF}, {0x1D00, 0x1DBF}, {0x1E00, 0x1FBC},
        {0x1FBE, 0x1FBE}, {0x1FC2, 0x1FCC}, {0x1FD0, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FFC}, {0x2071, 0x2071}, {0x207F, 0x207F}, {0x2090, 0x209C}, {0x2102, 0x2102},
        {0x2107, 0x2107}, {0x210A, 0x2113}, {0x2115, 0x2115}, {0x2119, 0x211D}, {0x2124, 0x2124}, {0x2126, 0x2126}, {0x2128, 0x2128}, {0x212A, 0x212D}, {0x212F, 0x2139},
        {0x213C, 0x213F}, {0x2145, 0x2149}, {0x214E, 0x214E}, {0x2C00, 0x2CE4}, {0x2CEB, 0x2CEE}, {0x2D00, 0x2D67}, {0x2D80, 0x2DDE}, {0x3005, 0x3006}, {0x3031, 0x3035},
        {0x303B, 0x303C}, {0x3041, 0x3096}, {0x309D, 0x309F}, {0x30A1, 0x30FA}, {0x30FC, 0x30FF}, {0x3105, 0x312F}, {0x3131, 0x318E}, {0x31A0, 0x31BF}, {0x31F0, 0x31FF},
        {0x3400, 0x4DBF}, {0x4E00, 0xA48C}, {0xA4D0, 0xA4FD}, {0xA500, 0xA60C}, {0xA610, 0xA61F}, {0xA640, 0xA66E}, {0xA680, 0xA69D}, {0xA717, 0xA71F}, {0xA722, 0xA788},
        {0xA78B, 0xA801}, {0xAC00, 0xD7A3}, {0xF900, 0xFAD9}, {0xFB00, 0xFB06}, {0xFB13, 0xFB17}, {0xFB1D, 0xFB1D}, {0xFB1F, 0xFB28}, {0xFB2A, 0xFBB1}, {0xFBD3, 0xFD3D},
        {0xFD50, 0xFDC7}, {0xFDF0, 0xFDFB}, {0xFE70, 0xFEFC}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A}, {0xFF66, 0xFFDC}, {0x10000, 0x100FA}, {0x10280, 0x1031F}, {0x10330, 0x10340},
        {0x10400, 0x1049D}, {0x1D400, 0x1D6C0}, {0x1D6C2, 0x1D6DA}, {0x1D6DC, 0x1D6FA}, {0x1D6FC, 0x1D714}, {0x1D716, 0x1D734}, {0x1D736, 0x1D74E}, {0x1D750, 0x1D76E},
        {0x1D770, 0x1D788}, {0x1D78A, 0x1D7A8}, {0x1D7AA, 0x1D7C2}, {0x1D7C4, 0x1D7CB}, {0x20000, 0x323AF}};
    return in_ranges(c, std::begin(r), std::end(r));
}

// How text is cut into words before merging
struct Splitter {
    enum class Kind { Whole, Gpt2, Llama3 };
    Kind kind = Kind::Gpt2;
    size_t max_digits = 3;  // Llama3: numbers are cut into groups of this many

    std::vector<std::string> split(const std::string& text) const {
        if (kind == Kind::Whole) return text.empty() ? std::vector<std::string>{} : std::vector<std::string>{text};
        std::vector<uint32_t> cp;
        std::vector<size_t> at;  // where each character starts
        for (size_t i = 0; i < text.size();) {
            size_t n;
            cp.push_back(next_char(text, i, &n));
            at.push_back(i);
            i += n;
        }
        at.push_back(text.size());
        std::vector<std::string> out;
        for (size_t i = 0; i < cp.size();) {
            size_t e = kind == Kind::Gpt2 ? gpt2(cp, i) : llama3(cp, i);
            out.push_back(text.substr(at[i], at[e] - at[i]));
            i = e;
        }
        return out;
    }

private:
    static bool other(uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); }
    template <typename F>
    static size_t run(const std::vector<uint32_t>& cp, size_t i, const F& f) {
        while (i < cp.size() && f(cp[i])) ++i;
        return i;
    }
    // The end of 's, 't, 're, 've, 'm, 'll or 'd at i, or i
    static size_t contraction(const std::vector<uint32_t>& cp, size_t i, bool any_case) {
        if (cp[i] != '\'' || i + 1 >= cp.size()) return i;
        auto lower = [&](size_t k) -> uint32_t {
            if (k >= cp.size()) return 0;
            return any_case && cp[k] >= 'A' && cp[k] <= 'Z' ? cp[k] + 32 : cp[k];
        };
        uint32_t a = lower(i + 1), b = lower(i + 2);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
        return i;
    }
    // \s+(?!\S)|\s+ : spaces, leaving the last one to the word that follows
    static size_t spaces(const std::vector<uint32_t>& cp, size_t i) {
        size_t e = run(cp, i, is_space);
        return e == cp.size() || e - i == 1 ? e : e - 1;
    }
    // '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    size_t gpt2(const std::vector<uint32_t>& cp, size_t i) const {
        size_t e = contraction(cp, i, false);
        if (e > i) return e;
        size_t j = cp[i] == ' ' ? i + 1 : i;
        if (j < cp.size()) {
            if (is_letter(cp[j])) return run(cp, j, is_letter);
            if (is_number(cp[j])) return run(cp, j, is_number);
            if (other(cp[j])) return run(cp, j, other);
        }
        return spaces(cp, i);
    }
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    size_t llama3(const std::vector<uint32_t>& cp, size_t i) const {
        size_t e = contraction(cp, i, true);
        if (e > i) return e;
        auto newline = [](uint32_t c) { return c == '\r' || c == '\n'; };
        if (is_letter(cp[i])) return run(cp, i, is_letter);
        if (!newline(cp[i]) && !is_number(cp[i]) && i + 1 < cp.size() && is_letter(cp[i + 1])) return run(cp, i + 1, is_letter);
        if (is_number(cp[i])) return std::min(run(cp, i, is_number), i + max_digits);
        size_t j = cp[i] == ' ' ? i + 1 : i;
        if (j < cp.size() && other(cp[j])) return run(cp, run(cp, j, other), newline);
        size_t end = run(cp, i, is_space);
        for (size_t k = end; k > i; --k) if (newline(cp[k - 1])) return k;
        return spaces(cp, i);
    }
};

// GPT-2's printable stand-in for every byte, as a character
inline const std::array<uint32_t, 256>& byte_chars() {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        uint32_t next = 256;
        for (uint32_t b = 0; b < 256; ++b) {
            bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            t[b] = printable ? b : next++;
        }
        return t;
    }();
    return table;
}
inline std::string bytes_to_chars(const std::string& bytes) {
    std::string s;
    for (char c : bytes) s += to_utf8(byte_chars()[(unsigned char)c]);
    return s;
}
// The bytes a token of a byte-level vocabulary stands for
inline std::string chars_to_bytes(const std::string& token) {
    static const std::unordered_map<uint32_t, unsigned char> back = [] {
        std::unordered_map<uint32_t, unsigned char> m;
        for (uint32_t b = 0; b < 256; ++b) m[byte_chars()[b]] = (unsigned char)b;
        return m;
    }();
    std::string s;
    for (size_t i = 0; i < token.size();) {
        size_t n;
        auto it = back.find(next_char(token, i, &n));
        if (it == back.end()) return token;  // not a byte-level token
        s += (char)it->second;
        i += n;
    }
    return s;
}

constexpr const char* kMetaspace = "\xE2\x96\x81";  // the lower block that stands for a space

inline std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (size_t i = s.find(from); i != std::string::npos; i = s.find(from, i + to.size())) s.replace(i, from.size(), to);
    return s;
}

}  // namespace tokenizer_detail

class Tokenizer {
public:
    // Ids of text. With special tokens, those written out in it ("<|end|>") are recognized, and the ones the
    // tokenizer puts around a text (a beginning-of-text token, say) are added.
    std::vector<int64_t> encode(const std::string& text, bool special = true) const {
        std::vector<int64_t> ids;
        if (special) ids.insert(ids.end(), prefix_.begin(), prefix_.end());
        WordCache cache;
        // Where each special token next occurs. A position is searched again only once the text before it
        // has been consumed, so every token's search moves forward through the text once.
        std::vector<const std::pair<const std::string, int64_t>*> specials;
        std::vector<size_t> next_at;
        if (special) {
            for (const auto& s : special_) {
                specials.push_back(&s);
                next_at.push_back(text.find(s.first));
            }
        }
        bool first = true;
        for (size_t i = 0; i < text.size();) {
            // The next special token, and the longest of those that start there
            size_t at = std::string::npos;
            const std::pair<const std::string, int64_t>* found = nullptr;
            for (size_t k = 0; k < specials.size(); ++k) {
                if (next_at[k] != std::string::npos && next_at[k] < i) next_at[k] = text.find(specials[k]->first, i);
                size_t p = next_at[k];
                if (p == std::string::npos) continue;
                if (p < at || (p == at && specials[k]->first.size() > found->first.size())) {
                    at = p;
                    found = specials[k];
                }
            }
            size_t end = found ? at : text.size();
            if (end > i) encode_plain(text.substr(i, end - i), first, ids, cache);
            first = false;
            if (!found) break;
            ids.push_back(found->second);
            i = end + found->first.size();
        }
        if (special) ids.insert(ids.end(), suffix_.begin(), suffix_.end());
        return ids;
    }

    // Text of ids; special tokens are left out unless asked for
    std::string decode(const std::vector<int64_t>& ids, bool special = false) const {
        std::string out;
        for (int64_t id : ids) {
            if (id < 0 || id >= (int64_t)tokens_.size()) throw std::out_of_range("Tokenizer: no token " + std::to_string(id));
            if (is_special_[(size_t)id]) {
                if (special) out += tokens_[(size_t)id];
                continue;
            }
            out += bytes_[(size_t)id];
        }
        if (metaspace_ && !out.empty() && out[0] == ' ') out.erase(0, 1);
        return out;
    }
    std::string decode(const std::shared_ptr<Tensor>& ids, bool special = false) const {
        std::vector<int64_t> v;
        for (float f : ids->to_vector()) v.push_back((int64_t)f);
        return decode(v, special);
    }

    int64_t vocab_size() const { return (int64_t)tokens_.size(); }
    // The id of a token as it is written in the vocabulary, or -1
    int64_t token_id(const std::string& token) const {
        auto s = special_.find(token);
        if (s != special_.end()) return s->second;
        auto it = vocab_.find(token);
        return it == vocab_.end() ? -1 : it->second;
    }
    // ids as a [1, time] tensor
    std::shared_ptr<Tensor> encode_tensor(const std::string& text, Device device = Device::CPU, bool special = true) const {
        auto ids = encode(text, special);
        std::vector<float> f(ids.begin(), ids.end());
        auto t = std::make_shared<Tensor>(std::vector<int64_t>{1, (int64_t)f.size()}, Device::CPU, DType::Float32, false);
        t->copy_from_vector(f);
        return device == Device::CPU ? t : t->to(device);
    }

    static Tokenizer from_file(const std::string& path) {
        auto j = json::parse_file(path);
        const auto& model = j.at("model");
        if (model.get_string("type", "BPE") != "BPE") throw std::runtime_error("Tokenizer: model type \"" + model.get_string("type", "") + "\" is not supported (BPE is)");
        Tokenizer t;
        t.read_rules(j);
        for (const auto& [token, id] : model.at("vocab").object) t.add_token(token, id.as_int(), false);
        if (const auto* added = j.find("added_tokens")) {
            for (const auto& a : added->array) {
                const std::string& content = a.at("content").as_string();
                t.add_token(content, a.at("id").as_int(), a.get_bool("special", false));
                t.special_[content] = a.at("id").as_int();
            }
        }
        t.ignore_merges_ = model.get_bool("ignore_merges", false);
        t.byte_fallback_ = model.get_bool("byte_fallback", false);
        if (const auto* unk = model.find("unk_token")) if (unk->is_string()) t.unk_ = t.token_id(unk->as_string());
        for (const auto& m : model.at("merges").array) {
            std::string a, b;
            if (m.is_string()) {
                size_t space = m.as_string().find(' ');
                if (space == std::string::npos) throw std::runtime_error("Tokenizer: bad merge");
                a = m.as_string().substr(0, space);
                b = m.as_string().substr(space + 1);
            } else {
                a = m.array.at(0).as_string();
                b = m.array.at(1).as_string();
            }
            t.add_merge(a, b);
        }
        t.read_template(j);
        t.finish();
        return t;
    }

    // Learns a byte-level vocabulary of vocab_size tokens: the 256 bytes, the most frequent merges, then the special
    // tokens
    static Tokenizer train(const std::string& text, int64_t vocab_size, const std::vector<std::string>& special_tokens = {}) {
        using namespace tokenizer_detail;
        Tokenizer t;
        for (int b = 0; b < 256; ++b) t.add_token(to_utf8(byte_chars()[(size_t)b]), b, false);
        int64_t merges = vocab_size - 256 - (int64_t)special_tokens.size();
        if (merges < 0) throw std::invalid_argument("Tokenizer::train: the vocabulary must hold 256 bytes and the special tokens");

        // Distinct words, as token ids, with their counts
        std::map<std::string, int64_t> counted;
        for (auto& w : t.splitter_.split(text)) ++counted[w];
        std::vector<std::vector<int>> words;
        std::vector<int64_t> counts;
        for (const auto& [w, n] : counted) {
            std::vector<int> ids;
            for (char c : w) ids.push_back((unsigned char)c);
            words.push_back(std::move(ids));
            counts.push_back(n);
        }
        using Pair = std::pair<int, int>;
        std::map<Pair, int64_t> pair_count;
        std::map<Pair, std::set<size_t>> where;  // words that (may) hold a pair
        for (size_t w = 0; w < words.size(); ++w) {
            for (size_t i = 0; i + 1 < words[w].size(); ++i) {
                Pair p{words[w][i], words[w][i + 1]};
                pair_count[p] += counts[w];
                where[p].insert(w);
            }
        }
        struct Entry {
            int64_t count;
            Pair pair;
            bool operator<(const Entry& o) const { return count != o.count ? count < o.count : pair > o.pair; }  // ties: the earlier pair
        };
        std::priority_queue<Entry> best;
        for (const auto& [p, n] : pair_count) best.push({n, p});
        for (int64_t m = 0; m < merges && !best.empty();) {
            Entry top = best.top();
            best.pop();
            if (pair_count[top.pair] != top.count) continue;  // counted again since
            if (top.count < 2) break;
            std::string a = t.tokens_[(size_t)top.pair.first], b = t.tokens_[(size_t)top.pair.second];
            // Another merge may spell the same token
            auto known = t.vocab_.find(a + b);
            int merged = known != t.vocab_.end() ? (int)known->second : (int)t.tokens_.size();
            if (known == t.vocab_.end()) {
                t.add_token(a + b, merged, false);
                ++m;
            }
            t.add_merge(a, b);
            std::set<Pair> changed;
            auto users = where[top.pair];
            for (size_t w : users) {
                auto& ids = words[w];
                std::vector<int> out;
                for (size_t i = 0; i < ids.size();) {
                    if (i + 1 < ids.size() && ids[i] == top.pair.first && ids[i + 1] == top.pair.second) {
                        out.push_back(merged);
                        i += 2;
                    } else {
                        out.push_back(ids[i++]);
                    }
                }
                for (size_t i = 0; i + 1 < ids.size(); ++i) {
                    Pair p{ids[i], ids[i + 1]};
                    pair_count[p] -= counts[w];
                    changed.insert(p);
                }
                for (size_t i = 0; i + 1 < out.size(); ++i) {
                    Pair p{out[i], out[i + 1]};
                    pair_count[p] += counts[w];
                    where[p].insert(w);
                    changed.insert(p);
                }
                ids = std::move(out);
            }
            for (const auto& p : changed) if (pair_count[p] > 0) best.push({pair_count[p], p});
        }
        for (const auto& s : special_tokens) {
            int64_t id = (int64_t)t.tokens_.size();
            t.add_token(s, id, true);
            t.special_[s] = id;
        }
        t.finish();
        return t;
    }

    // As a tokenizer.json. For tokenizers made by train.
    void save(const std::string& path) const {
        if (metaspace_ || splitter_.kind != tokenizer_detail::Splitter::Kind::Gpt2) throw std::runtime_error("Tokenizer::save: only trained tokenizers can be saved");
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Tokenizer::save: cannot write " + path);
        out << "{\"version\": \"1.0\", \"truncation\": null, \"padding\": null, \"added_tokens\": [";
        bool comma = false;
        std::map<int64_t, std::string> specials;
        for (const auto& s : special_) specials[s.second] = s.first;
        for (const auto& [id, s] : specials) {
            out << (comma ? ", " : "") << "{\"id\": " << id << ", \"content\": " << json::quote(s)
                << ", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}";
            comma = true;
        }
        out << "], \"normalizer\": null, \"pre_tokenizer\": {\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true, \"use_regex\": true}, "
            << "\"post_processor\": null, \"decoder\": {\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true, \"use_regex\": true}, "
            << "\"model\": {\"type\": \"BPE\", \"dropout\": null, \"unk_token\": null, \"continuing_subword_prefix\": null, \"end_of_word_suffix\": null, "
            << "\"fuse_unk\": false, \"byte_fallback\": false, \"vocab\": {";
        comma = false;
        for (size_t i = 0; i < tokens_.size(); ++i) {
            if (is_special_[i]) continue;
            out << (comma ? ", " : "") << json::quote(tokens_[i]) << ": " << i;
            comma = true;
        }
        out << "}, \"merges\": [";
        for (size_t m = 0; m < merge_list_.size(); ++m) out << (m ? ", " : "") << json::quote(merge_list_[m].first + " " + merge_list_[m].second);
        out << "]}}\n";
    }

private:
    using Splitter = tokenizer_detail::Splitter;

    void add_token(const std::string& token, int64_t id, bool special) {
        if (id < 0 || id > (1 << 24)) throw std::runtime_error("Tokenizer: token id out of range");
        if ((size_t)id >= tokens_.size()) {
            tokens_.resize((size_t)id + 1);
            is_special_.resize((size_t)id + 1, 0);
        }
        tokens_[(size_t)id] = token;
        is_special_[(size_t)id] = special ? 1 : 0;
        vocab_[token] = id;
    }
    void add_merge(const std::string& a, const std::string& b) {
        auto ia = vocab_.find(a), ib = vocab_.find(b), im = vocab_.find(a + b);
        if (ia == vocab_.end() || ib == vocab_.end() || im == vocab_.end()) throw std::runtime_error("Tokenizer: a merge uses tokens the vocabulary lacks");
        merges_.emplace(key(ia->second, ib->second), std::make_pair((int64_t)merge_list_.size(), im->second));
        merge_list_.emplace_back(a, b);
    }
    static uint64_t key(int64_t a, int64_t b) { return ((uint64_t)a << 32) | (uint64_t)b; }

    // What each token decodes to
    void finish() {
        bytes_.clear();
        for (size_t i = 0; i < tokens_.size(); ++i) {
            const auto& tok = tokens_[i];
            if (special_.count(tok)) {
                bytes_.push_back(tok);
            } else if (metaspace_) {
                unsigned byte = 0;
                if (tok.size() == 6 && tok.compare(0, 3, "<0x") == 0 && tok[5] == '>' && std::sscanf(tok.c_str(), "<0x%02X>", &byte) == 1) bytes_.push_back(std::string(1, (char)byte));
                else bytes_.push_back(tokenizer_detail::replace_all(tok, tokenizer_detail::kMetaspace, " "));
            } else {
                bytes_.push_back(tokenizer_detail::chars_to_bytes(tok));
            }
        }
    }

    // The word-splitting rules, from the normalizer and the pre-tokenizer
    void read_rules(const json::Value& j) {
        using namespace tokenizer_detail;
        std::vector<const json::Value*> steps;
        std::function<void(const json::Value*)> collect = [&](const json::Value* v) {
            if (!v || !v->is_object()) return;
            for (const char* list : {"normalizers", "pretokenizers"}) {
                if (const auto* inner = v->find(list)) {
                    for (const auto& s : inner->array) collect(&s);
                    return;
                }
            }
            steps.push_back(v);
        };
        collect(j.find("normalizer"));
        collect(j.find("pre_tokenizer"));
        bool byte_level = false, split = false;
        for (const auto* s : steps) {
            const std::string type = s->get_string("type", "");
            if (type == "ByteLevel") {
                byte_level = true;
                if (s->get_bool("add_prefix_space", false)) prefix_space_ = true;
                if (s->get_bool("use_regex", true) && !split) {
                    splitter_.kind = Splitter::Kind::Gpt2;
                    split = true;
                }
            } else if (type == "Split") {
                const auto& pattern = s->at("pattern");
                const std::string regex = pattern.get_string("Regex", "");
                if (regex.find("[^\\r\\n\\p{L}\\p{N}]?\\p{L}+") == std::string::npos) throw std::runtime_error("Tokenizer: the splitting rule \"" + regex + "\" is not supported");
                splitter_.kind = Splitter::Kind::Llama3;
                splitter_.max_digits = regex.find("\\p{N}{1,3}") != std::string::npos ? 3 : 1;
                split = true;
            } else if (type == "Metaspace") {
                metaspace_ = true;
                const std::string scheme = s->get_string("prepend_scheme", s->get_bool("add_prefix_space", true) ? "always" : "never");
                prepend_ = scheme == "never" ? Prepend::Never : scheme == "first" ? Prepend::First : Prepend::Always;
            } else if (type == "Prepend") {
                metaspace_ = true;
                prepend_ = Prepend::Always;
            } else if (type == "Replace") {
                metaspace_ = metaspace_ || s->get_string("content", "") == kMetaspace;
            } else if (type != "NFC" && type != "NFKC" && type != "") {
                throw std::runtime_error("Tokenizer: the step \"" + type + "\" is not supported");
            }
        }
        if (metaspace_) splitter_.kind = Splitter::Kind::Whole;
        else if (!byte_level) throw std::runtime_error("Tokenizer: only byte-level and sentencepiece-style BPE are supported");
    }

    // Special tokens around a text
    void read_template(const json::Value& j) {
        const json::Value* p = j.find("post_processor");
        if (!p || !p->is_object()) return;
        std::vector<const json::Value*> all{p};
        if (const auto* inner = p->find("processors")) {
            all.clear();
            for (const auto& v : inner->array) all.push_back(&v);
        }
        for (const auto* v : all) {
            if (v->get_string("type", "") != "TemplateProcessing") continue;
            bool after = false;
            const json::Value* named = v->find("special_tokens");  // name -> {"ids": [...]}, when given
            for (const auto& part : v->at("single").array) {
                if (part.has("Sequence")) {
                    after = true;
                    continue;
                }
                const auto* s = part.find("SpecialToken");
                if (!s) continue;
                const std::string& name = s->at("id").as_string();
                auto& out = after ? suffix_ : prefix_;
                const json::Value* entry = named && named->is_object() ? named->find(name) : nullptr;
                const json::Value* listed = entry ? entry->find("ids") : nullptr;
                std::vector<int64_t> found;
                if (listed && listed->is_array()) {
                    for (const auto& id : listed->array) found.push_back(id.as_int());
                } else {
                    found.push_back(token_id(name));
                }
                for (int64_t id : found) {
                    if (id < 0 || id >= (int64_t)tokens_.size()) {
                        throw std::runtime_error("Tokenizer: the post_processor adds \"" + name + "\", which is not in the vocabulary");
                    }
                    out.push_back(id);
                }
            }
        }
    }

    // The ids of words already encoded during one call of encode: text repeats words, and merging is the
    // costly part. Bounded, so that text of few repeats cannot grow it without end.
    struct WordCache {
        static constexpr size_t kMaxWords = 1 << 16;
        std::unordered_map<std::string, std::vector<int64_t>> ids;
    };

    void encode_plain(const std::string& text_in, bool first, std::vector<int64_t>& ids, WordCache& cache) const {
        using namespace tokenizer_detail;
        std::string text = text_in;
        if (metaspace_) {
            text = replace_all(text, " ", kMetaspace);
            if (prepend_ == Prepend::Always || (prepend_ == Prepend::First && first)) text = kMetaspace + text;
        } else if (prefix_space_ && text[0] != ' ') {
            text = " " + text;
        }
        for (const auto& word : splitter_.split(text)) {
            auto hit = cache.ids.find(word);
            if (hit != cache.ids.end()) {
                ids.insert(ids.end(), hit->second.begin(), hit->second.end());
                continue;
            }
            size_t start = ids.size();
            encode_word(word, ids);
            if (cache.ids.size() < WordCache::kMaxWords) cache.ids.emplace(word, std::vector<int64_t>(ids.begin() + (std::ptrdiff_t)start, ids.end()));
        }
    }

    void encode_word(const std::string& word, std::vector<int64_t>& ids) const {
        using namespace tokenizer_detail;
        if (ignore_merges_) {
            auto it = vocab_.find(metaspace_ ? word : bytes_to_chars(word));
            if (it != vocab_.end()) {
                ids.push_back(it->second);
                return;
            }
        }
        // The word's smallest pieces
        std::vector<int64_t> s;
        if (metaspace_) {
            for (size_t i = 0; i < word.size();) {
                size_t n;
                next_char(word, i, &n);
                auto it = vocab_.find(word.substr(i, n));
                if (it != vocab_.end()) {
                    s.push_back(it->second);
                } else if (byte_fallback_) {
                    for (size_t k = 0; k < n; ++k) {
                        char name[8];
                        std::snprintf(name, sizeof(name), "<0x%02X>", (unsigned)(unsigned char)word[i + k]);
                        auto b = vocab_.find(name);
                        s.push_back(b != vocab_.end() ? b->second : unk_);
                    }
                } else {
                    s.push_back(unk_);
                }
                i += n;
            }
        } else {
            for (char c : word) {
                auto it = vocab_.find(to_utf8(byte_chars()[(unsigned char)c]));
                s.push_back(it != vocab_.end() ? it->second : unk_);
            }
        }
        // The earliest-learned merge first, leftmost first among equals; pieces form a linked list
        struct Candidate {
            int64_t rank;
            size_t left;
            int64_t a, b, merged;
            bool operator<(const Candidate& o) const { return rank != o.rank ? rank > o.rank : left > o.left; }
        };
        const size_t none = std::numeric_limits<size_t>::max();
        size_t n = s.size();
        std::vector<size_t> prev(n), next(n);
        for (size_t i = 0; i < n; ++i) {
            prev[i] = i == 0 ? none : i - 1;
            next[i] = i + 1 < n ? i + 1 : none;
        }
        std::priority_queue<Candidate> queue;
        auto consider = [&](size_t i) {
            if (i == none || next[i] == none) return;
            auto it = merges_.find(key(s[i], s[next[i]]));
            if (it != merges_.end()) queue.push({it->second.first, i, s[i], s[next[i]], it->second.second});
        };
        for (size_t i = 0; i < n; ++i) consider(i);
        std::vector<char> gone(n, 0);
        while (!queue.empty()) {
            Candidate c = queue.top();
            queue.pop();
            size_t i = c.left;
            if (gone[i] || next[i] == none || s[i] != c.a || s[next[i]] != c.b) continue;  // merged away since
            size_t r = next[i];
            gone[r] = 1;
            s[i] = c.merged;
            next[i] = next[r];
            if (next[i] != none) prev[next[i]] = i;
            consider(i);
            consider(prev[i]);
        }
        for (size_t i = 0; i < n; ++i) {
            if (gone[i]) continue;
            if (s[i] < 0) throw std::runtime_error("Tokenizer: the text has a character the vocabulary cannot spell");
            ids.push_back(s[i]);
        }
    }

    enum class Prepend { Never, First, Always };
    std::vector<std::string> tokens_;  // as written in the vocabulary
    std::vector<std::string> bytes_;   // what they decode to
    std::vector<char> is_special_;
    std::unordered_map<std::string, int64_t> vocab_;
    std::map<std::string, int64_t> special_;  // recognized in text before anything else
    std::unordered_map<uint64_t, std::pair<int64_t, int64_t>> merges_;  // pair -> (rank, result)
    std::vector<std::pair<std::string, std::string>> merge_list_;
    std::vector<int64_t> prefix_, suffix_;
    Splitter splitter_;
    bool metaspace_ = false, ignore_merges_ = false, byte_fallback_ = false, prefix_space_ = false;
    Prepend prepend_ = Prepend::Never;
    int64_t unk_ = -1;
};
