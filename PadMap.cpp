#include "PadMap.h"
#include <string>
#include <sstream>

namespace mcopt
{
    int PadMap::CalculateHash(const int cobo, const int asad, const int aget, const int channel)
    {
        return channel + aget*100 + asad*10000 + cobo*1000000;
    }

    pad_t PadMap::find(const int cobo, const int asad, const int aget, const int channel) const
    {
        auto hash = CalculateHash(cobo, asad, aget, channel);
        auto foundItem = table.find(hash);
        if (foundItem != table.end()) {
            return foundItem->second;
        }
        else {
            return missingValue; // an invalid value
        }
    }

    Address PadMap::reverseFind(const pad_t pad) const
    {
        auto foundItem = reverseTable.find(pad);
        if (foundItem != reverseTable.end()) {
            return foundItem->second;
        }
        else {
            return Address {missingValue, missingValue, missingValue, missingValue}; // an invalid value
        }
    }

    void PadMap::insert(const int cobo, const int asad, const int aget, const int channel, const pad_t pad)
    {
        auto hash = CalculateHash(cobo, asad, aget, channel);

        table.emplace(hash, pad);
        reverseTable.emplace(pad, Address{cobo, asad, aget, channel});
        return;
    }

    bool PadMap::empty() const
    {
        return table.empty();
    }

    PadMap::PadMap(const std::string& path)
    {
        std::ifstream file (path, std::ios::in|std::ios::binary);

        // MUST throw out the first two junk lines in file. No headers!

        if (!file.good()) throw 0; // FIX THIS!

        if (table.size() != 0) {
            table.clear();
        }

        std::string line;

        while (!file.eof()) {
            int cobo, asad, aget, channel;
            pad_t pad;
            getline(file,line,'\n');
            std::stringstream lineStream(line);
            std::string element;

            getline(lineStream, element,',');
            if (element == "-1" || element == "") continue; // KLUDGE!
            cobo = stoi(element);

            getline(lineStream, element,',');
            asad = stoi(element);

            getline(lineStream, element,',');
            aget = stoi(element);

            getline(lineStream, element,',');
            channel = stoi(element);

            getline(lineStream, element);
            pad = stoi(element);

            insert(cobo, asad, aget, channel, pad);
        }
    }
}
