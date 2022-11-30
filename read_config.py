import yaml

def read_yaml(config_name=''):
    with open(config_name, "r", encoding='UTF-8') as stream:
        config = yaml.safe_load(stream)

    return config


if __name__ == '__main__':

    config_name = 'config_template.yaml'
    config = read_yaml(config_name)
    print(config)
