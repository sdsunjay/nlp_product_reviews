
def main():
    """Main function of the program."""
    # Specify path
    training_filepath = 'data/clean_training1.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()
    df = read_data(training_filepath)
    # import pdb; pdb.set_trace()
    df = df.sample(frac=0.5).reset_index(drop=True)
    df = df.dropna()
    
    # split into training, validation, and test sets
    training, test = np.array_split(df.head(60000), 2)
    labels = df['human_tag']   
    del df['target']
    train = data_utils.TensorDataset(torch.Tensor(np.array(df)), torch.Tensor(np.array(labels)))
    train_loader = data_utils.DataLoader(train, batch_size = 1000, shuffle = True, num_workers=4)
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    features = []
    for i, batch in enumerate(train_loader):   
    	# When we have more time
    	# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-large-uncased') 
    	features.append(tokenizeText1(batch, 'clean_text', model_class, tokenizer_class, pretrained_weights))

    testing_filepath = 'data/clean_testing1.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(testing_filepath)
    if(isExist):
        print('Reading from ' + testing_filepath)
    else:
        print('Testing file not found in the app path.')
        exit()
    df1 = read_data(testing_filepath)
    df1 = df1.dropna()
    a = np.array_split(df1,8)
    i = 0
    values = []
    for aa in a: 
        output_name = str(i) 
        print('Run: ' + output_name)
        i += 1
        testing_features  = tokenizeText1(aa, 'clean_text', model_class, tokenizer_class, pretrained_weights)
        final_y_pred = trainClassifiers(features, labels, testing_features)
        values = np.concatenate((values, final_y_pred), axis=0)

    df1["human_tag"] = values
    header = ["ID", "human_tag"]
    output_path = 'result/MLP100' 
    print('Output: ' + output_path)
    df1.to_csv(output_path, columns = header)
    
    # features = tokenizeText2(df, 'clean_text', model_class)
    # features  = tokenizeText2(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)
    # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)

if __name__ == "__main__":
    main()
