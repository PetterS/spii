// Petter Strandmark 2014.
#ifndef SPII_SOLVER_CALLBACKS_H
#define SPII_SOLVER_H

#include <fstream>

#include <spii/solver.h>

namespace spii {

class FileCallback
{
public:
	FileCallback(std::ofstream& file_)
		: file(file_)
	{ }

	bool operator()(const CallbackInformation& information) const
	{
		for (std::size_t i = 0; i < information.x->size(); ++i) {
			file << (*information.x)[i] << " ";
		}
		file << std::endl;
		return true;
	}

private:
	std::ofstream& file;
};

}  // namespace spii

#endif
