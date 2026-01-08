import csv
import torch
import random
from param import parameter_parser

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

def data_pro(args):
    dataset = dict()
    dataset['cd_p'] = read_csv(args.dataset_path + '/a.csv')
    dataset['cd_true'] = read_csv(args.dataset_path + '/a.csv')
    print(dataset['cd_p'].shape)
    zero_index = []
    one_index = []
    for i in range(dataset['cd_p'].size(0)):
        for j in range(dataset['cd_p'].size(1)):
            if dataset['cd_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['cd_p'][i][j] >= 1:
                one_index.append([i, j])
    print(len(zero_index))
    print(len(one_index))
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = torch.LongTensor(zero_index)
    one_tensor = torch.LongTensor(one_index)

    dataset['one_index'] = one_tensor
    dataset['zero_index'] = zero_tensor

    dataset['cd'] = dict()
    dss_matrix = read_csv(args.dataset_path + '/DSS.csv')
    dss_edge_index = get_edge_index(dss_matrix)
    dataset['dss'] = {'data_matrix': dss_matrix, 'edges': dss_edge_index}

    css_matrix = read_csv(args.dataset_path + '/GSS.csv')
    css_edge_index = get_edge_index(css_matrix)
    print(css_edge_index.shape)
    dataset['css'] = {'data_matrix': css_matrix, 'edges': css_edge_index}
    dataset['df'] = dss_matrix
    dataset['cf'] = css_matrix
    return dataset

def main():
    args = parameter_parser()
    dataset = data_pro(args)
if __name__ == "__main__":
    main()
