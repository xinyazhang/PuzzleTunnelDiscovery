#include "quickgl.h"
#include "osr_rt_texture.h"
#include <stdexcept>

static_assert(sizeof(GLuint) == sizeof(unsigned int), "RtTexture::tex_ is not matching GLuint");

namespace osr {

struct RtTexture::Private {
	GLuint tex = 0;
	GLint ifmt;
	GLint efmt;
	GLint etype;
	const void* zero_valued_buffer;
};

const GLint BYTE_IFMT[] = { GL_R8, GL_RG8, GL_RGB8, GL_RGBA8 };
const GLint BYTE_EFMT[] = { GL_RED, GL_RG, GL_RGB, GL_RGBA };
const uint8_t BYTE_ZERO[] = { 0, 0, 0, 0 };

const GLint FP_IFMT[] = { GL_R32F, GL_RG32F, GL_RGB32F, GL_RGBA32F };
const GLint FP_EFMT[] = { GL_RED, GL_RG, GL_RGB, GL_RGBA };
const float FP_ZERO[] = { 0.0, 0.0, 0.0, 0.0 };

const GLint INT_IFMT[] = { GL_R32I, GL_RG32I, GL_RGB32I, GL_RGBA32I };
const GLint INT_EFMT[] = { GL_RED_INTEGER, GL_RG_INTEGER, GL_RGB_INTEGER, GL_RGBA_INTEGER };
const GLint INT_ZERO[] = { 0, 0, 0, 0 };

RtTexture::RtTexture()
{
}

RtTexture::~RtTexture()
{
	if (p_)
		glDeleteTextures(1, &p_->tex);
}

void RtTexture::ensure(int w, int h, int nchannel, int type)
{
	if (p_)
		return;
	const GLint *fmt;
	const GLint *efmt;
	GLint etype;
	const void *zvbuf;
	switch (type) {
		case BYTE_TYPE:
			fmt = BYTE_IFMT;
			efmt = BYTE_EFMT;
			etype = GL_UNSIGNED_BYTE;
			zvbuf = BYTE_ZERO;
			break;
		case FLOAT_TYPE:
			fmt = FP_IFMT;
			efmt = FP_EFMT;
			etype = GL_FLOAT;
			zvbuf = FP_ZERO;
			break;
		case INT32_TYPE:
			fmt = INT_IFMT;
			efmt = INT_EFMT;
			etype = GL_INT;
			zvbuf = INT_ZERO;
			break;
		default:
			throw std::runtime_error("Unknown type " + std::to_string(type));
	}
	p_ = std::make_unique<Private>();
	p_->ifmt = fmt[nchannel];
	p_->efmt = efmt[nchannel];
	p_->etype = etype;
	p_->zero_valued_buffer = zvbuf;

	CHECK_GL_ERROR(glGenTextures(1, &p_->tex));
	CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, p_->tex));
	CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, p_->ifmt, w, h, 0,
				    p_->efmt, p_->etype, 0));
	CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

void RtTexture::clear(const void* overrided_value)
{
	if (!p_)
		return ;
	const void* data = p_->zero_valued_buffer;
	if (overrided_value)
		data = overrided_value;
	CHECK_GL_ERROR(glClearTexImage(p_->tex, 0, p_->efmt, p_->etype, data));
}

unsigned int RtTexture::getTex() const
{
	return p_ ? p_->tex : 0;
}

FrameBuffer::FrameBuffer()
{
}

FrameBuffer::~FrameBuffer()
{
}

void FrameBuffer::attachRt(int shader_location, RtTexturePtr rt)
{
	if (rt_locactions_[shader_location] == rt)
		return ;
	rt_locactions_[shader_location] = rt;
	dirty_ = true;
}

void FrameBuffer::detachRt(int shader_location)
{
	if (!rt_locactions_[shader_location])
		return ;
	rt_locactions_[shader_location].reset();
	dirty_ = true;
}

void FrameBuffer::create(int w, int h)
{
	if (fb_)
		return ;
	CHECK_GL_ERROR(glGenFramebuffers(1, &fb_));
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, fb_));

	CHECK_GL_ERROR(glGenRenderbuffers(1, &depth_component_));
	CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, depth_component_));
	CHECK_GL_ERROR(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h));
	CHECK_GL_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_component_));
}


void FrameBuffer::activate()
{
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, fb_));
	if (dirty_) {
		int max_loc = 0;
		for (const auto& kv : rt_locactions_) {
			if (!kv.second)
				continue;
			max_loc = std::max(kv.first, max_loc);
		}
		draws_.resize(max_loc + 1, GL_NONE);
		// std::cerr << "max_loc: " << max_loc << std::endl;
		int attid = 0;
		for (const auto& kv : rt_locactions_) {
			auto att_enum = GL_COLOR_ATTACHMENT0 + attid;
			GLuint tex_id = 0;
			if (kv.second) { // RtTexturePtr may be reset
				tex_id = kv.second->getTex();
				// std::cerr << "draws_[" << kv.first << "] = " << attid << std::endl;
				draws_[kv.first] = att_enum;
			}
			CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, att_enum, tex_id, 0));
			attid++;
		}
		CHECK_GL_ERROR(glDrawBuffers(draws_.size(), draws_.data()));
		dirty_ = false;
	}
}

void FrameBuffer::deactivate()
{
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}
void FrameBuffer::readShaderLocation(int shader_location)
{
	CHECK_GL_ERROR(glReadBuffer(draws_[shader_location]));
}

}
