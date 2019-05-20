from keras.datasets import reuters

def main():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    print(len(train_data))
    print(len(train_labels))
    print(len(test_data))
    print(len(test_labels))
    pass
if __name__ == '__main__':
    main()