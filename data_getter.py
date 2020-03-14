import os
import os.path as path
import csv
import pickle
import numpy as np

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/Data')
    base_dir = os.getcwd()

    file_name_delim = "_"
    in_file_delim = " "

    for experiment, subdir in enumerate(next(os.walk(os.getcwd()))[1]):

        str_parameters = subdir.split("_")[1:]

        parameters = []
        for str in str_parameters:
            parameters.append(float(str))

        # print(parameters)

        disp_file_name = file_name_delim.join(["Disp"] + str_parameters)
        react_file_name = file_name_delim.join(["React"] + str_parameters)

        data_path = base_dir + "/" + subdir + "/Displ_React/"

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

        particle_path = base_dir + "/" + subdir + "/Inputs/"

        data_files = [f for f in os.listdir(particle_path) if path.isfile(path.join(particle_path, f))]

        particle_data_list = [None for i in range(len(data_files))]
        particle_data_times = [None for i in range(len(data_files))]
        for i, f in enumerate(data_files):

            info = f.split('_')

            particle_data_times[i] = info[0]

            num_water_particles = int(info[1])

            num_beam_particles = int(info[2])

            num_static_particles = int(info[3][:-7])

            with open(path.join(particle_path, f), "rb") as data_file:

                data = np.array(pickle.load(data_file), order='F')

            data.resize((data.shape[0], data.shape[1] + 4))

            data[:, 3:-3] = data[:, 2:-4]

            data[:, 2] = 0.0

            data[:num_water_particles, -3:] = [1, 0, 0]

            data[num_water_particles:num_water_particles + num_beam_particles, -3:] = [0, 1, 0]

            data[num_water_particles + num_beam_particles:, -3:] = [0, 0, 1]

            particle_data_list[i] = data.copy()


        sorted_indices = np.argsort(particle_data_times)

        particle_data_times = np.array(sorted(particle_data_times))

        particle_data = np.array(particle_data_list)[sorted_indices]

        data_file_name = file_name_delim.join(["data"] + str_parameters) + ".npz"

        np.savez(data_file_name, inputs=inputs, outputs=outputs, points=particle_data, times=particle_data_times)

        del particle_data_times, particle_data, inputs, outputs
