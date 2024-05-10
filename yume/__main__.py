import argparse
# from .yume import Yume

def main():
    parser = argparse.ArgumentParser(description='Yume CLI')
    subparsers = parser.add_subparsers(dest='command')

    download_parser = subparsers.add_parser('download', help='Download the model')
    download_parser.add_argument('--model_name', type=str, required=True, help='Name of the model to download')

    query_parser = subparsers.add_parser('query', help='Run a query on the model')
    query_parser.add_argument('query_text', type=str, help='Text to query the model with')

    args = parser.parse_args()

    if args.command == 'download':
        # yume = Yume()
        # yume.download_model(args.model_name)
        result = args.model_name
        print(result)
    elif args.command == 'query':
        # yume = Yume()
        # result = yume.query(args.query_text)
        result = args.query_text
        print(result)

if __name__ == "__main__":
    main()