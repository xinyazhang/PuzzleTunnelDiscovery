#if GPU_ENABLED

#ifndef OSR_RT_TEXTURE_H
#define OSR_RT_TEXTURE_H

#include <memory>
#include <map>
#include <vector>

namespace osr {

// Render-target Texture
class RtTexture {
public:
	enum {
		BYTE_TYPE = 0,
		FLOAT_TYPE = 1,
		INT32_TYPE = 2
	};

	RtTexture();
	~RtTexture();

	void ensure(int w, int h, int nchannel, int type);
	void clear(const void* overrided_value  = nullptr); // Should be quiet when tex_ == 0

	unsigned int getTex() const;
private:
	struct Private;
	std::unique_ptr<Private> p_;
};

using RtTexturePtr = std::shared_ptr<RtTexture>;

class FrameBuffer {
public:
	FrameBuffer();
	~FrameBuffer();

	// Will not take effect until next call to activate()
	void attachRt(int shader_location, RtTexturePtr);
	void detachRt(int shader_location);

	void create(int w, int h);
	void activate();
	void deactivate();
	void readShaderLocation(int);

	unsigned int getFb() const { return fb_; }
private:
	unsigned int fb_ = 0;
	unsigned int depth_component_ = 0;

	std::map<int, RtTexturePtr> rt_locactions_;
	std::vector<unsigned int> draws_;
	bool dirty_ = false;
};

}

#endif

#endif // GPU_ENABLED
