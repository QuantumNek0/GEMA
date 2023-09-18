import streamlit as st
from streamlit_pills import pills
import gema


def main():
    st.title("GEMA")
    st.write("Melody generator")
    key_root = pills("Key Root", ["C", "D", "E", "F", "G", "A", "B"])
    key_type = pills("Type", ["natural", "#", "b"])
    key_mode = pills("Mode", ["maj", "min"])
    bits_per_note = pills("Bits per Note", ["4", "8", "16"])
    population_size = pills("Population size", ["4", "6", "8", "10"])
    bpm = pills("Beats per Minute", ["80", "120", "150"])
    output_size = pills("Output Size", ["1", "2", "3"])
    first_gen_method = pills("First Generation Method", ["Random", "VAE"])

    if st.button("Generate Melody"):
        gema.main(key_root, key_type, key_mode,
                  int(bits_per_note), int(population_size),
                  int(output_size), int(bpm), first_gen_method)


if __name__ == '__main__':
    main()

# Terminal command:
# streamlit run path/GEMA/app.py
