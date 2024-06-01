import streamlit as st
from yume import Yume
from yume.config import yume_small


def main():
    st.set_page_config(page_title="夢の生成器", page_icon="🌙", layout="wide")

    st.title("夢の生成器")
    st.write("Welcome to our Dream Generator! 🌙")

    st.sidebar.header("Select a Dream Theme")
    theme = st.sidebar.selectbox(
        "Choose a theme", ["Fantasy", "Adventure", "Romance", "Horror"]
    )

    st.sidebar.header("Customize Your Dream")
    prompt = st.sidebar.text_input("Enter a prompt or keyword")
    num_sentences = st.sidebar.slider("Number of sentences to generate", 1, 10, 5)
    temperature = st.sidebar.slider("Creativity of LLM (Temperature)", 0.1, 1.0, 0.5)

    generate_button = st.sidebar.button("Generate Dream")

    st.header("Your Dream")

    if generate_button:

        with st.spinner("Generating dream..."):

            dream_text = generate_dream_text(theme, prompt, num_sentences, temperature)

            st.text_area("", value=dream_text, height=600)
    else:
        st.text_area("", height=600, placeholder="Your dream will appear here...")


def generate_dream_text(theme, prompt, num_sentences, temperature):

    yume = Yume(config=yume_small)

    yume.load_pretrained("zaibutcooler/yume")

    full_prompt = f"{theme} - {prompt}"

    # Generate text using the Yume model
    return yume.sample(full_prompt)


if __name__ == "__main__":
    main()
