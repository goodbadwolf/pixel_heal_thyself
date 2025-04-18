"""
termlog.py - Enhanced tee-like utility for terminal streams.
This module provides functionality similar to the Unix 'tee' command but with
enhanced handling of terminal control characters. It reads from stdin, displays
the content on the terminal (stdout), and simultaneously writes a processed
version to a specified output file.
The main enhancement over standard 'tee' is the handling of carriage returns (\r)
which are commonly used for updating in-place terminal output (like progress bars).
When writing to the output file, the module intelligently processes these control
characters to produce a clean, readable log.
Usage:
    python termlog.py OUTPUT_FILE
    Program will read from stdin until EOF or keyboard interrupt.
Example:
    command_with_progress_output | python termlog.py logfile.txt
"""

import sys


def process_stream(input_stream, output_file, stdout):
    """Process input stream character-by-character, handle control characters and write to file."""

    buffer = []

    while True:
        char = input_stream.read(1)

        if not char:  # End of input
            break

        # Write raw output to stdout for terminal display
        stdout.write(char)
        stdout.flush()

        # Process control characters for the file output
        if char == "\r":
            # Clear buffer when carriage return is encountered
            buffer = []
        else:
            buffer.append(char)
            if char == "\n":
                # Write the current buffer contents to file
                output_file.write("".join(buffer))
                output_file.flush()
                buffer = []

    # Write any remaining content in the buffer
    if buffer:
        output_file.write("".join(buffer))
        output_file.flush()


def main():
    """Main entry point for the tee+ utility."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} OUTPUT_FILE")
        sys.exit(1)

    output_filename = sys.argv[1]

    try:
        with open(output_filename, "w", encoding="utf-8") as output_file:
            process_stream(sys.stdin, output_file, sys.stdout)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
