#ifndef MCOPT_PAD_MAP_H
#define MCOPT_PAD_MAP_H

#include <iostream>
#include <fstream>
#include <unordered_map>

namespace mcopt
{
    typedef uint16_t pad_t;

    struct Address
    {
        int cobo;
        int asad;
        int aget;
        int channel;
    };

    class PadMap
    {
    public:
        PadMap() = default;
        PadMap(const std::string& path);

        void insert(const int cobo, const int asad, const int aget, const int channel, const pad_t pad);

        pad_t find(const int cobo, const int asad, const int aget, const int channel) const;
        Address reverseFind(const pad_t pad) const;

        bool empty() const;

        pad_t missingValue {20000};

    protected:
        static int CalculateHash(const int cobo, const int asad, const int aget, const int channel);

        std::unordered_map<int,pad_t> table;  // The hashtable, maps hash:value
        std::unordered_map<pad_t,Address> reverseTable;
    };
}

#endif /* defined(MCOPT_PAD_MAP_H) */
