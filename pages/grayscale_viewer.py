import numpy as np
import pandas as pd
import streamlit as st
from keras.datasets import fashion_mnist, mnist


DATASET_OPTIONS = {
	"MNIST": {
		"loader": mnist.load_data,
		"class_names": [str(index) for index in range(10)],
	},
	"Fashion MNIST": {
		"loader": fashion_mnist.load_data,
		"class_names": [
			"T-shirt/top",
			"Trouser",
			"Pullover",
			"Dress",
			"Coat",
			"Sandal",
			"Shirt",
			"Sneaker",
			"Bag",
			"Ankle boot",
		],
	},
}


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str):
	(x_train, y_train), _ = DATASET_OPTIONS[dataset_name]["loader"]()
	return x_train, y_train


def initialize_state() -> None:
	defaults = {
		"viewer_dataset": "MNIST",
		"viewer_class_index": 0,
		"viewer_sample_index": None,
		"viewer_mode": "image",
	}
	for key, value in defaults.items():
		if key not in st.session_state:
			st.session_state[key] = value


def choose_random_sample(x_data: np.ndarray, y_data: np.ndarray, class_index: int) -> int:
	matching_indices = np.flatnonzero(y_data == class_index)
	if len(matching_indices) == 0:
		raise ValueError("선택한 클래스에 해당하는 이미지가 없습니다.")
	return int(np.random.choice(matching_indices))


def reset_sample_if_filter_changed(dataset_name: str, class_index: int) -> None:
	changed = (
		dataset_name != st.session_state.viewer_dataset
		or class_index != st.session_state.viewer_class_index
	)
	if changed:
		st.session_state.viewer_dataset = dataset_name
		st.session_state.viewer_class_index = class_index
		st.session_state.viewer_sample_index = None
		st.session_state.viewer_mode = "image"


def show_image_view(image: np.ndarray) -> None:
	st.image(image, clamp=True, width=280)


def show_matrix_view(image: np.ndarray) -> None:
	matrix_df = pd.DataFrame(image.astype(int))
	st.dataframe(matrix_df, use_container_width=True, height=520)


initialize_state()

st.title("MNIST / Fashion MNIST 이미지 변환기")
st.write("클래스를 고르면 해당 조건의 임의 이미지를 보여주고, 버튼으로 28x28 숫자 배열로 전환할 수 있습니다.")

st.sidebar.header("데이터 설정")
dataset_name = st.sidebar.selectbox(
	"데이터셋 선택",
	list(DATASET_OPTIONS.keys()),
)

class_names = DATASET_OPTIONS[dataset_name]["class_names"]
selected_class_name = st.sidebar.selectbox("클래스 선택", class_names)
class_index = class_names.index(selected_class_name)

reset_sample_if_filter_changed(dataset_name, class_index)

x_train, y_train = load_dataset(dataset_name)

if st.session_state.viewer_sample_index is None:
	st.session_state.viewer_sample_index = choose_random_sample(x_train, y_train, class_index)

selected_index = st.session_state.viewer_sample_index
selected_image = x_train[selected_index]

sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
if sidebar_col1.button("다른 이미지", use_container_width=True):
	st.session_state.viewer_sample_index = choose_random_sample(x_train, y_train, class_index)
	st.session_state.viewer_mode = "image"
	st.rerun()

if sidebar_col2.button("초기화", use_container_width=True):
	st.session_state.viewer_mode = "image"
	st.rerun()

st.sidebar.markdown("---")
st.sidebar.write(f"선택 데이터셋: {dataset_name}")
st.sidebar.write(f"선택 클래스: {selected_class_name}")
st.sidebar.write(f"샘플 인덱스: {st.session_state.viewer_sample_index}")

st.subheader("선택된 샘플")
info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("데이터셋", dataset_name)
info_col2.metric("클래스", selected_class_name)
info_col3.metric("배열 크기", f"{selected_image.shape[0]} x {selected_image.shape[1]}")

if st.session_state.viewer_mode == "image":
	show_image_view(selected_image)
else:
	show_matrix_view(selected_image)

button_label = "이미지를 숫자로 전환" if st.session_state.viewer_mode == "image" else "숫자를 이미지로 전환"
if st.button(button_label, type="primary", use_container_width=True):
	st.session_state.viewer_mode = "matrix" if st.session_state.viewer_mode == "image" else "image"
	st.rerun()

with st.expander("28x28 숫자 배열 안내"):
	st.write("각 칸의 값은 픽셀 밝기입니다. 0에 가까울수록 검은색, 255에 가까울수록 흰색입니다.")
	st.write("MNIST와 Fashion MNIST 모두 28x28 크기의 흑백 이미지를 사용합니다.")
