#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <exception>
#include <string>

class TrackingFailed : public std::exception
{
public:
    TrackingFailed(std::string m) : msg(m) {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg;
};

class GasError : public std::exception
{
public:
    GasError(std::string m) : msg(m) {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg;
};

#endif /* end of include guard: EXCEPTION_H */
