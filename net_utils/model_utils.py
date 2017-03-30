from StringIO import StringIO



def extract_features(model, image_dataset, selected_layer=None, batch_size=32, verbose=False):
    # type: (Model, ImageDataset, basestring, int, bool, bool) -> ImageDataset
    def printv(s):
        if verbose: print(s)

    if selected_layer is  not None:
        model = submodel(model, output_layer_name=selected_layer)


    printv("Starting feed-forward on network (prediction)...")
    prediction = model.predict(image_dataset.data, batch_size, verbose)
    feature_vectors = image_dataset
    feature_vectors.load_ndarray(prediction)
    # feature_vectors.save_hdf5(h5_out_features_dataset_path)
    printv("Done.")
    return feature_vectors



def submodel( original_model, input_layer_name=None, output_layer_name=None ):
    # Type: (Model, basestring, basestring) -> Model
    from keras.engine import Model

    inp = original_model.input
    outp = original_model.output
    if input_layer_name is not None:
        inp = original_model.get_layer(input_layer_name).output
    if output_layer_name is not None:
        outp = original_model.get_layer(output_layer_name).output
    return Model(input=inp, output=outp)



def save_to_json(model, file_name, file_mode='w'):
    # Type: (Model, basestring, basestring) -> None
    f = open(file_name, file_mode)
    f.write(model.to_json())
    f.close()


def summary_on_file(model, file_name, file_mode='w'):
    # Type: (Model, basestring, basestring) -> None
    import sys
    with open(file_name, file_mode) as f:
        def_stdout = sys.stdout
        sys.stdout = f
        model.summary(line_length=200)
        sys.stdout = def_stdout


def summary(model):
    # Type: (Model) -> basestring
    import sys
    def_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    model.summary(line_length=200)
    s = "".join(mystdout.buflist)
    sys.stdout = def_stdout
    return s


def summary_small(model):
    # Type: (Model) -> basestring
    lnl = layer_name_list(model)
    s = ""
    for ln in lnl:
        s += ln + "\n"
    return s

def layer_name_list(model):
    # Type: (Model) -> basestring
    lnl = []
    for layer in model.layers:
        lnl.append(layer.name)
    return lnl

def layer_name_list_print(model):
    print summary_small(model)