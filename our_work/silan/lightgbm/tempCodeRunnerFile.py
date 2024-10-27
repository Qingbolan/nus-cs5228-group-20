
    #     if scaler is None:
    #         scaler = StandardScaler()
    #         X[columns_to_standardize] = pd.DataFrame(scaler.fit_transform(X[columns_to_standardize]), 
    #                                                  columns=columns_to_standardize, 
    #                                                  index=X.index)
    #     else:
    #         X[columns_to_standardize] = pd.DataFrame(scaler.transform(X[columns_to_standardize]), 
    #                                                  columns=columns_to_standardize, 
    #                                                  index=X.index)