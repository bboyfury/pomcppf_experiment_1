import sys

def hello_world(name):
    print(f"Hello, World! {name}")

# Check if the user provided a command-line argument
if len(sys.argv) > 1:
    # Get the argument from the command line
    user_input = sys.argv[1]
    # Call the function with the command-line argument
    hello_world(user_input)
else:
    print("Please provide a name as a command-line argument.")
