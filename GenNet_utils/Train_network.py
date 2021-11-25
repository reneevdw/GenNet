model = None
masks = None
datapath = args.path
jobid = args.ID
lr_opt = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
l1_value = args.L1
problem_type = args.problem_type

if args.genotype_path == "undefined":
    genotype_path = datapath
else:
    genotype_path = args.genotype_path

check_data(datapath=datapath, genotype_path=genotype_path, mode=problem_type)

optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)

train_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 1)
val_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 2)
test_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 3)
inputsize = get_inputsize(genotype_path)

folder, resultpath = get_paths(jobid)

print("jobid =  " + str(jobid))
print("folder = " + str(folder))
print("batchsize = " + str(batch_size))
print("lr = " + str(lr_opt))

if os.path.exists(datapath + "/topology.csv"):
    model, masks = create_network_from_csv(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                           l1_value=l1_value)
if len(glob.glob(datapath + "/*.npz")) > 0:
    model, masks = create_network_from_npz(datapath=datapath, inputsize=inputsize, genotype_path=genotype_path,
                                           l1_value=l1_value)

model.compile(loss="mse", optimizer=optimizer_model,
              metrics=["mse"])

with open(resultpath + '/model_architecture.txt', 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

earlystop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto',
                                      restore_best_weights=True)
saveBestModel = K.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                            verbose=1, save_best_only=True, mode='auto')

# %%
if os.path.exists(resultpath + '/bestweights_job.h5'):
    print('Model already Trained')
else:
    history = model.fit_generator(
        generator=TrainDataGenerator(datapath=datapath,
                                     genotype_path=genotype_path,
                                     batch_size=batch_size,
                                     trainsize=int(train_size)),
        shuffle=True,
        epochs=epochs,
        verbose=1,
        callbacks=[earlystop, saveBestModel],
        workers=15,
        use_multiprocessing=True,
        validation_data=EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=batch_size,
                                      setsize=val_size, inputsiz=inputsize, evalset="validation")
    )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(resultpath + "train_val_loss.png")
    plt.show()

model.load_weights(resultpath + '/bestweights_job.h5')
print("Finished")
print("Analysis over the validation set")
pval = model.predict_generator(
    EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=1, setsize=val_size,
                  evalset="validation", inputsize=inputsize))
yval = get_labels(datapath, set_number=2)
fig, mse_val, explained_variance_val, r2_val = evaluate_performance_regression(yval, pval)
np.save(resultpath + "/pval.npy", pval)
fig.savefig(resultpath + "/validation_predictions.png", bbox_inches='tight', pad_inches=0)

print("Analysis over the test set")
ptest = model.predict_generator(
    EvalGenerator(datapath=datapath, genotype_path=genotype_path, batch_size=1, setsize=test_size, inputsize=inputsize,
                  evalset="test"))
ytest = get_labels(datapath, set_number=3)
fig, mse_test, explained_variance_test, r2_test = evaluate_performance_regression(ytest, ptest)
np.save(resultpath + "/ptest.npy", ptest)
fig.savefig(resultpath + "/test_predictions.png", bbox_inches='tight', pad_inches=0)

# %%

with open(resultpath + '/Results_' + str(jobid) + '.txt', 'a') as f:
    f.write('\n Jobid = ' + str(jobid))
    f.write('\n Batchsize = ' + str(batch_size))
    f.write('\n Learningrate = ' + str(lr_opt))
    f.write('\n Optimizer = ' + str(optimizer_model))
    f.write('\n L1 value = ' + str(l1_value))
    f.write('\n')
    f.write("Validation set")
    f.write('\n Mean squared error = ' + str(mse_val))
    f.write('\n Explained variance = ' + str(explained_variance_val))
    f.write('\n R2 = ' + str(r2_val))
    f.write("Test set")
    f.write('\n Mean squared error = ' + str(mse_test))
    f.write('\n Explained variance = ' + str(explained_variance_val))
    f.write('\n R2 = ' + str(r2_test))

if os.path.exists(datapath + "/topology.csv"):
    importance_csv = create_importance_csv(datapath, model, masks)
    importance_csv.to_csv(resultpath + "connection_weights.csv")
