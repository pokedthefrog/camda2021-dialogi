from dialogi import dialogi, classifiers


if __name__ == '__main__':
    dlg = dialogi.Dialogi()

    # Use suffix 'public' to load pre-computed publication data.
    dlg.load_pubtator(sfx='public')           # Pubtator annots.
    dlg.load_proctxts(sfx='public')           # Processed texts.
    dlg.load_concvecs(sfx='public_reduced')   # Concept vectors.

    # Concatenate Chemical and Disease spaces into one.
    dlg.concat_spaces(('MeanVecs_Disease', 'MeanVecs_Chemical'))
    for ds_name in ['pos', 'neg', 'val0', 'val1']:
        del dlg.data['vecs'][ds_name].data['MeanVecs']   # Delete unused spaces.

    # Create inner and outer folds for Nested Cross-Validation.
    ncv_folds = dlg.create_cvfolds(num_outer_folds=10, num_inner_folds=5)

    # Create a 'Classifiers' object and start training, validation, and testing.
    lstm_model_obj = classifiers.LSTMClassifier()
    ncv_results = dlg.train_test_model(ncv_folds, lstm_model_obj)
