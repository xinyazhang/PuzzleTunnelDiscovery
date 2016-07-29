#ifndef READ_HPP
#define READ_HPP

#include <istream>

void skip_spaces_and_comments(std::istream& fin);

template<typename T>
T read(std::istream& fin)
{
	T ret;
	skip_spaces_and_comments(fin);
	fin >> ret;
	return ret;
}


#endif
