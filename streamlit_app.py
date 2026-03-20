import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist

st.set_page_config(page_title="인공신경망 샘플 앱", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name):
    if dataset_name == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        class_names = [str(i) for i in range(10)]
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    return x_train, y_train, x_test, y_test, class_names


def preprocess_data(x_train, x_test):
    x_train_ann = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test_ann = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    return x_train_ann, x_test_ann


def build_model(hidden_units, learning_rate):
    tf.keras.backend.clear_session()

    layers = [tf.keras.layers.Input(shape=(28 * 28,))]
    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# 사이드바: 데이터셋 선택
st.sidebar.title("데이터셋 선택")
dataset_name = st.sidebar.selectbox(
    "데이터셋을 선택하세요",
    ["MNIST", "Fashion-MNIST"]
)

# 데이터 로드
x_train, y_train, x_test, y_test, class_names = load_dataset(dataset_name)

# 사이드바: 원본 이미지 미리보기
st.sidebar.subheader("원본 이미지 미리보기")
img_idx = st.sidebar.slider("이미지 인덱스", 0, len(x_train) - 1, 0)

fig_preview, ax_preview = plt.subplots()
ax_preview.imshow(x_train[img_idx], cmap="gray")
ax_preview.axis("off")
ax_preview.set_title(f"Label: {class_names[int(y_train[img_idx])]}")
st.sidebar.pyplot(fig_preview)
plt.close(fig_preview)

st.sidebar.markdown("---")
st.sidebar.subheader("데이터셋 분할 정보")
st.sidebar.markdown(
    """
**Train 데이터**: 모델이 학습하는 데 사용하는 데이터 (전체의 약 81%)  
**Validation 데이터**: 학습 중 모델의 성능을 검증하는 데 사용하는 데이터 (전체의 약 9%)  
**Test 데이터**: 학습이 끝난 후 모델의 최종 성능을 평가하는 데 사용하는 데이터 (전체의 약 10%)
"""
)

# 메인 페이지
st.title("인공신경망 샘플 앱")
st.write("데이터셋:", dataset_name)

st.header("인공신경망 조건 설정")

num_hidden = int(st.number_input("은닉층의 수", min_value=1, max_value=3, value=1, step=1))

hidden_units = []
for i in range(num_hidden):
    units = int(
        st.number_input(
            f"은닉층 {i+1}의 노드 수",
            min_value=8,
            max_value=128,
            value=32,
            step=8,
            key=f"hidden_{i}"
        )
    )
    hidden_units.append(units)

epochs = int(
    st.number_input(
        "에포크 수 (전체 데이터셋을 몇 번 반복)",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )
)

learning_rate = st.number_input(
    "학습률",
    min_value=0.0001,
    max_value=0.1,
    value=0.001,
    format="%.4f"
)

train_btn = st.button("학습 시작")

if train_btn:
    with st.spinner("학습 중입니다..."):
        x_train_ann, x_test_ann = preprocess_data(x_train, x_test)
        model = build_model(hidden_units, learning_rate)

        log_area = st.empty()
        val_accs = []

        for epoch in range(epochs):
            history = model.fit(
                x_train_ann,
                y_train,
                epochs=1,
                validation_split=0.1,
                verbose=0,
                batch_size=128
            )
            val_acc = history.history["val_accuracy"][-1]
            val_accs.append(val_acc)
            log_area.text(f"에포크 {epoch+1}/{epochs} - 검증 정확도: {val_acc:.4f}")

        test_loss, test_acc = model.evaluate(x_test_ann, y_test, verbose=0)

        st.success("학습 완료!")
        st.write(f"최종 검증 정확도: {val_accs[-1]:.4f}")

        st.subheader("테스트 데이터 결과")
        st.write(f"테스트 정확도: {test_acc:.4f}")

        st.write("테스트 데이터 예측 결과 샘플")
        preds = np.argmax(model.predict(x_test_ann[:20], verbose=0), axis=1)

        fig_result, axes = plt.subplots(2, 10, figsize=(15, 4))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(x_test[i], cmap="gray")
            ax.axis("off")
            ax.set_title(
                f"GT:{class_names[int(y_test[i])]}\nPR:{class_names[int(preds[i])]}",
                fontsize=8
            )

        st.pyplot(fig_result)
        plt.close(fig_result)