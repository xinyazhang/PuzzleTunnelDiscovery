#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>

using namespace std;

vector<string> write_blend_rules(const vector<string>& meshes, int tier, bool remeshing = true)
{
	vector<string> ret;
	int i;
	for(i = 0; i + 1 < meshes.size(); i += 2) {
		string prefix = "Tier-"+to_string(tier)+"-Merge-"+to_string(i/2)+"-";
		string out = prefix+"done.obj";
		string fn;

		if (remeshing)
			fn = prefix+"re.obj";
		else
			fn = out;
		cout << fn << ": " << meshes[i] << " " << meshes[i+1] << endl;
		cout << "\t$(BIN)/blend -NI -prefix " << prefix << " $?" << endl;
		if (remeshing)
			cout << "\tfix_mesh.py " << out << " " << fn << endl; // Assumes PyMesh has been installed and available in $PATH
		cout << endl;
		ret.emplace_back(fn);
	}
	for(;i < meshes.size(); i++)
		ret.emplace_back(meshes[i]);
	return ret;
}

int main(int argc, char* argv[])
{
	string binary_path;
	vector<string> meshes;
	int opt = 0;
	bool remeshing = true;
	while ((opt = getopt(argc, argv, "nb:")) != -1) {
		switch (opt) {
			case 'n':
				remeshing = false;
				break;
			case 'b':
				binary_path = optarg;
				break;
		};
	}
	for(int i = optind; i < argc; i++) {
		meshes.emplace_back(argv[i]);
	}
	cout << "BIN='" << binary_path << "'" << endl << endl;
	cout << "first: done.obj" << endl << endl;
	int tier = 0;
	while (meshes.size() > 1) {
		meshes = write_blend_rules(meshes, tier, remeshing);
		tier++;
	}
	cout << "done.obj: " << meshes.front() << endl;
	cout << "\tcp " << meshes.front() << " done.obj" << endl;
}
