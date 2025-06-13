def datasetInfo(dataset_name):
    if dataset_name == "Cars":
        nb_classes  = 196
    elif dataset_name == "Aircraft":
        nb_classes  = 100
    elif dataset_name == "CUB200":
        nb_classes  = 200
    elif dataset_name == "Flower102":
        nb_classes  = 102
    elif dataset_name == "NABird":
        nb_classes  = 555
    elif dataset_name == "Cigarette":
        nb_classes  = 6  # 6 cigarette brands: 云烟99, 云烟105, 云烟87, 云烟116, 湘烟7号, 中川208
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets are : Cars, Aircraft, CUB200, Flower102, NABird and Cigarette")

    # Add more elif statements for other datasets as needed

    return nb_classes
