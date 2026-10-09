#include <copylib/utils.hpp>

#include <iostream>
#include <sstream>

namespace copylib::utils {

void dump_to_cerr(std::string_view msg) { std::cerr << msg; }
void dump_to_cout(std::string_view msg) { std::cout << msg; }

std::vector<std::string> split(const std::string& str, char delim) {
	std::vector<std::string> result;
	std::string token;
	std::istringstream token_stream(str);
	while(std::getline(token_stream, token, delim)) {
		result.push_back(token);
	}
	return result;
}

int64_t parse_command_line_option(int argc, char** argv, const std::string& option, int64_t default_value) {
	for(int i = 1; i < argc - 1; i++) {
		if(std::string(argv[i]) == option) { return std::stoll(argv[i + 1]); }
	}
	return default_value;
}

} // namespace copylib::utils
