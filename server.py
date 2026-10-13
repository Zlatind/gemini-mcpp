import os
from mcp.server.fastmcp import FastMCP, Image
from google import genai

mcp = FastMCP("gemini-images", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")


@mcp.tool()
def generate_image(prompt: str) -> Image:
    """Generate an image with Gemini from a text prompt (English works best)."""
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    for part in resp.candidates[0].content.parts:
        if part.inline_data:
            return Image(data=part.inline_data.data, format="png")
    raise ValueError("Gemini did not return an image.")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
