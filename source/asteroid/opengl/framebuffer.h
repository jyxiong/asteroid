#pragma once

namespace Asteroid {

	struct FramebufferSpecification
	{
		uint32_t Width, Height;
	};

	class Framebuffer
	{
	public:
		Framebuffer(const FramebufferSpecification& spec);
		~Framebuffer();

		void Invalidate();

	    void Bind();
		void Unbind();

		uint32_t GetColorAttachmentRendererID() const { return m_ColorAttachment; }

	private:
		uint32_t m_RendererID;
		uint32_t m_ColorAttachment;

		FramebufferSpecification m_Specification;
	};

}