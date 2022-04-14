def metrics_from_txt(path_to_txt):
    path_to_txt = "/opt/data/team/hien/models/model_01_03_100_0.txt"
    
    
    #extract AP and AR from .txt, which was generated while training a model
    

    training = []
    evaluation = []
    metric = []
    with open(path_to_txt, 'r') as content:
        for line in content:
            if line.startswith("Epoch:"):
                training.append(line)
            if line.startswith("Test:"):
                evaluation.append(line)
            if line.startswith(" Average "):  
                metric.append(line)


    print(training)
    print(evaluation)
    print(metric)

    with open('listfile.txt', 'w') as filehandle:
        for listitem in metric:
            filehandle.write("%s"% listitem)
