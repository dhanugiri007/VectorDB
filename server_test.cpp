#include "httplib.h"
#include <iostream>

int main() {
    httplib::Server svr;

    svr.Get("/hello", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("Hello from C++ server!", "text/plain");
    });

    std::cout << "Server starting on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}