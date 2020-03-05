import os
import csv
import pickle

os.chdir('C:\\Users\\reidc\\OneDrive\\Documents\\current_classes\\cs535\\project\\Data')
base_dir = os.getcwd()

file_name_delim = "_"
in_file_delim = " "

for subdir in next(os.walk(os.getcwd()))[1]:

    str_parameters = subdir.split("_")[1:]

    parameters = []
    for str in str_parameters:
        parameters.append(float(str))

    print(parameters)

    disp_file_name = file_name_delim.join(["Disp"] + str_parameters)
    react_file_name = file_name_delim.join(["React"] + str_parameters)

    data_path = base_dir + "\\" + subdir + "\\Displ_React\\"

    inputs = []
    outputs = []

    with open(data_path + disp_file_name, "r") as disp_file:

        reader = csv.reader(disp_file, delimiter=in_file_delim, quoting=csv.QUOTE_NONNUMERIC)

        for row in reader:

            inputs.append(parameters + [row[0]])
            outputs.append([row[1]])

    with open(data_path + react_file_name, "r") as react_file:

        reader = csv.reader(react_file, delimiter=in_file_delim, quoting=csv.QUOTE_NONNUMERIC)

        for i, row in enumerate(reader):

            if row[0] != inputs[i][-1]:
                raise Exception("Shit's not right")

            outputs[i] += row[1:]

    pickle_file_name = file_name_delim.join(["data"] + str_parameters) + ".pickle"

    with open(pickle_file_name, "wb") as pickle_file:

        pickle.dump([inputs, outputs], pickle_file)

