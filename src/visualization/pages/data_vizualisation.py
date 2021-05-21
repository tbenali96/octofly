import streamlit as st


def write():

    st.header("DATA VISUALIZATION AND STATISTICS PAGE")

    st.subheader("1. Flight details")
    user_infos = create_text_inputs()

    st.subheader("2. Some Statistics")
    video_upload = st.file_uploader("Choose a video...", type=["mp4", "mpeg", "avi", "mov"])

    output_folder = get_output_folder(user_infos)
    print(video_upload)

    if video_upload is not None:
        video_filepath = download_video(video_upload)
        fps_model, size_ratio, confidence = create_sidebar(video_filepath)
        video_infos = get_video_infos(video_filepath=video_filepath)
        display_infos_sidebar(video_infos)

        st.subheader("3. Launch model")
        if st.button("Run"):
            create_folder(output_folder)
            st.markdown(f"output folder: {output_folder}")
            params_infos = get_params_infos(
                                    output_folder=output_folder,
                                    fps_model=fps_model,
                                    size_ratio=size_ratio,
                                    confidence=confidence)
            pre_run_infos = {**video_infos, **params_infos, **user_infos}
            run_infos = run_model(pre_run_infos)
            infos = {**pre_run_infos, **run_infos}
            save_infos(infos)


if __name__ == "__main__":
    write()