import streamlit as st

st.set_page_config(page_title="Quick Test", page_icon="✅", layout="centered")

st.title("✅ Streamlit Render Test")
st.write("If you can see this, Streamlit is rendering correctly.")

with st.expander("Details"):
    st.write({
        "session_initialized": bool(st.session_state),
        "session_keys": list(st.session_state.keys()),
    })

if st.button("Click me"):
    st.success("Button works!")


