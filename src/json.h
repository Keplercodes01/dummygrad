#pragma once
// Reading JSON: model configs, checkpoint headers and indexes.
//
//   auto config = json::parse_file("config.json");
//   int64_t layers = config.at("num_hidden_layers").as_int();
//   double theta = config.get("rope_theta", 10000.0);      // default when missing or null

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace json {

class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::pair<std::string, Value>> object;  // in the order of the file

    bool is_null() const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }

    // The member, or null when there is none
    const Value* find(const std::string& key) const {
        for (const auto& member : object) if (member.first == key) return &member.second;
        return nullptr;
    }
    bool has(const std::string& key) const { return find(key) != nullptr; }
    const Value& at(const std::string& key) const {
        const Value* v = find(key);
        if (!v) throw std::runtime_error("json: no member \"" + key + "\"");
        return *v;
    }

    double as_number() const {
        expect(Type::Number, "a number");
        return number;
    }
    int64_t as_int() const {
        double d = as_number();
        if (d != std::floor(d) || std::fabs(d) > 9007199254740992.0) throw std::runtime_error("json: expected an integer");
        return (int64_t)d;
    }
    bool as_bool() const {
        expect(Type::Bool, "true or false");
        return boolean;
    }
    const std::string& as_string() const {
        expect(Type::String, "a string");
        return string;
    }

    double get(const std::string& key, double fallback) const {
        const Value* v = find(key);
        return v && !v->is_null() ? v->as_number() : fallback;
    }
    int64_t get_int(const std::string& key, int64_t fallback) const {
        const Value* v = find(key);
        return v && !v->is_null() ? v->as_int() : fallback;
    }
    bool get_bool(const std::string& key, bool fallback) const {
        const Value* v = find(key);
        return v && !v->is_null() ? v->as_bool() : fallback;
    }
    std::string get_string(const std::string& key, const std::string& fallback) const {
        const Value* v = find(key);
        return v && !v->is_null() ? v->as_string() : fallback;
    }

private:
    void expect(Type t, const char* what) const {
        if (type != t) throw std::runtime_error(std::string("json: expected ") + what);
    }
};

class Parser {
public:
    Parser(const char* text, size_t size) : p_(text), begin_(text), end_(text + size) {}

    Value parse_document() {
        Value v = parse_value(0);
        skip_space();
        if (p_ != end_) fail("trailing characters");
        return v;
    }

private:
    const char* p_;
    const char* begin_;
    const char* end_;

    [[noreturn]] void fail(const std::string& what) const { throw std::runtime_error("json: " + what + " at byte " + std::to_string(p_ - begin_)); }
    void skip_space() {
        while (p_ != end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }
    char peek() {
        skip_space();
        if (p_ == end_) fail("unexpected end");
        return *p_;
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++p_;
    }
    bool literal(const char* word) {
        const char* q = p_;
        for (; *word; ++word, ++q) if (q == end_ || *q != *word) return false;
        p_ = q;
        return true;
    }

    Value parse_value(int depth) {
        if (depth > 256) fail("nested too deeply");
        Value v;
        char c = peek();
        if (c == '{') {
            v.type = Value::Type::Object;
            ++p_;
            if (peek() == '}') {
                ++p_;
                return v;
            }
            for (;;) {
                if (peek() != '"') fail("expected a member name");
                std::string key = parse_string();
                expect(':');
                v.object.emplace_back(std::move(key), parse_value(depth + 1));
                char next = peek();
                ++p_;
                if (next == '}') return v;
                if (next != ',') fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type = Value::Type::Array;
            ++p_;
            if (peek() == ']') {
                ++p_;
                return v;
            }
            for (;;) {
                v.array.push_back(parse_value(depth + 1));
                char next = peek();
                ++p_;
                if (next == ']') return v;
                if (next != ',') fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.type = Value::Type::String;
            v.string = parse_string();
            return v;
        }
        if (literal("true") || literal("false")) {
            v.type = Value::Type::Bool;
            v.boolean = c == 't';
            return v;
        }
        if (literal("null")) return v;
        v.type = Value::Type::Number;
        v.number = parse_number();
        return v;
    }

    double parse_number() {
        const char* q = p_;
        auto digits = [&] {
            const char* from = q;
            while (q != end_ && *q >= '0' && *q <= '9') ++q;
            return q != from;
        };
        if (q != end_ && *q == '-') ++q;
        if (!digits()) fail("expected a value");
        if (q != end_ && *q == '.') {
            ++q;
            if (!digits()) fail("expected digits");
        }
        if (q != end_ && (*q == 'e' || *q == 'E')) {
            ++q;
            if (q != end_ && (*q == '+' || *q == '-')) ++q;
            if (!digits()) fail("expected digits");
        }
        double d = std::strtod(std::string(p_, q).c_str(), nullptr);
        p_ = q;
        return d;
    }

    unsigned parse_hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i, ++p_) {
            if (p_ == end_) fail("unexpected end");
            char c = *p_;
            unsigned d = c >= '0' && c <= '9' ? (unsigned)(c - '0') : c >= 'a' && c <= 'f' ? (unsigned)(c - 'a' + 10) : c >= 'A' && c <= 'F' ? (unsigned)(c - 'A' + 10) : 99u;
            if (d == 99u) fail("bad \\u escape");
            v = v * 16 + d;
        }
        return v;
    }

    static void append_utf8(std::string& s, unsigned cp) {
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
    }

    std::string parse_string() {
        std::string s;
        ++p_;  // the opening quote
        for (;;) {
            if (p_ == end_) fail("unterminated string");
            char c = *p_++;
            if (c == '"') return s;
            if (c != '\\') {
                s += c;
                continue;
            }
            if (p_ == end_) fail("unterminated string");
            char e = *p_++;
            switch (e) {
                case '"': case '\\': case '/': s += e; break;
                case 'b': s += '\b'; break;
                case 'f': s += '\f'; break;
                case 'n': s += '\n'; break;
                case 'r': s += '\r'; break;
                case 't': s += '\t'; break;
                case 'u': {
                    unsigned cp = parse_hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {  // the first half of a pair
                        if (end_ - p_ < 2 || p_[0] != '\\' || p_[1] != 'u') fail("unpaired surrogate");
                        p_ += 2;
                        unsigned low = parse_hex4();
                        if (low < 0xDC00 || low >= 0xE000) fail("unpaired surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(s, cp);
                    break;
                }
                default: fail("unknown escape");
            }
        }
    }
};

inline Value parse(const std::string& text) { return Parser(text.data(), text.size()).parse_document(); }

inline Value parse_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("json: cannot open " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(text);
}

// text as a JSON string, quotes included
inline std::string quote(const std::string& text) {
    static const char* hex = "0123456789abcdef";
    std::string s = "\"";
    for (char c : text) {
        unsigned char u = (unsigned char)c;
        if (c == '"' || c == '\\') {
            s += '\\';
            s += c;
        } else if (u < 0x20) {
            s += "\\u00";
            s += hex[u >> 4];
            s += hex[u & 15];
        } else {
            s += c;
        }
    }
    return s + "\"";
}

}  // namespace json
