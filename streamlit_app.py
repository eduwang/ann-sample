import streamlit as st
# 데이터셋 로딩 함수
from tensorflow.keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

# 사이드바: 데이터셋 선택 드롭다운
st.sidebar.title('데이터셋 선택')
dataset_name = st.sidebar.selectbox(
    '데이터셋을 선택하세요',
    ['MNIST', 'Fashion-MNIST']
)

# 데이터셋 로딩
if dataset_name == 'MNIST':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    class_names = [str(i) for i in range(10)]
elif dataset_name == 'Fashion-MNIST':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 사이드바: 원본 이미지 미리보기
st.sidebar.subheader('원본 이미지 미리보기')
img_idx = st.sidebar.slider('이미지 인덱스', 0, len(x_train)-1, 0)
fig, ax = plt.subplots()
ax.imshow(x_train[img_idx], cmap='gray')
ax.axis('off')
ax.set_title(f"Label: {class_names[y_train[img_idx]]}")
st.sidebar.pyplot(fig)

# Main page: ANN 조건 설정 및 학습
st.title('인공신경망 샘플 앱')
st.write('데이터셋:', dataset_name)


st.header('인공신경망 조건 설정')
# 은닉층 수 설정
num_hidden = st.number_input('은닉층의 수', min_value=1, max_value=3, value=1)
# 각 은닉층별 노드 수 설정
hidden_units = []
for i in range(num_hidden):
    units = st.number_input(f'은닉층 {i+1}의 노드 수', min_value=8, max_value=128, value=32, step=8, key=f"hidden_{i}")
    hidden_units.append(units)
# 에포크 설명 추가
epochs = st.number_input('에포크 수 (전체 데이터셋을 몇 번 반복)', min_value=1, max_value=50, value=5)
# 러닝레이트 → 학습률
learning_rate = st.number_input('학습률', min_value=0.0001, max_value=0.1, value=0.001, format='%f')

train_btn = st.button('학습 시작')

if train_btn:
    st.info('학습 중...')
    # 데이터 전처리
    x_train_ann = x_train.reshape(-1, 28*28) / 255.0
    x_test_ann = x_test.reshape(-1, 28*28) / 255.0

    import tensorflow as tf
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(28*28,)))
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 학습 로그 표시
    log_area = st.empty()
    val_accs = []
    for epoch in range(epochs):
        history = model.fit(x_train_ann, y_train, epochs=1, validation_split=0.1, verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        val_accs.append(val_acc)
        log_area.text(f"에포크 {epoch+1}/{epochs} - 검증 정확도: {val_acc:.4f}")

    st.success('학습 완료!')
    st.write(f"최종 검증 정확도: {val_accs[-1]:.4f}")

    # 테스트 데이터 예측 및 정확도
    test_loss, test_acc = model.evaluate(x_test_ann, y_test, verbose=0)
    st.subheader('테스트 데이터 결과')
    st.write(f"테스트 정확도: {test_acc:.4f}")

    # 예측 결과 시각화
    st.write('테스트 데이터 예측 결과 샘플')
    preds = np.argmax(model.predict(x_test_ann[:20]), axis=1)
    fig2, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_test[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f"GT:{class_names[y_test[i]]}\nPR:{class_names[preds[i]]}", fontsize=8)
    st.pyplot(fig2)

