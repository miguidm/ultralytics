#!/usr/bin/env python3
"""
Example usage of the SSH Crawler module
Demonstrates how to use the ssh_crawler function with configurable hostname and port
"""

from ssh_crawler import ssh_crawler, display_tables


def example_with_password():
    """Example: Connect using password authentication"""
    print("Example 1: Using Password Authentication")
    print("-" * 50)

    hostname = "localhost"
    port = 22
    username = "myuser"
    password = "mypass"

    file_list = ssh_crawler(
        hostname=hostname,
        port=port,
        username=username,
        password=password
    )

    display_tables(file_list)


def example_with_ssh_key():
    """Example: Connect using SSH key authentication"""
    print("\nExample 2: Using SSH Key Authentication")
    print("-" * 50)

    hostname = "192.168.1.100"
    port = 22
    username = "myuser"
    key_filename = "/home/user/.ssh/id_rsa"

    file_list = ssh_crawler(
        hostname=hostname,
        port=port,
        username=username,
        key_filename=key_filename
    )

    display_tables(file_list)


def example_with_custom_port():
    """Example: Connect to SSH server on custom port"""
    print("\nExample 3: Using Custom Port")
    print("-" * 50)

    hostname = "example.com"
    port = 2222
    username = "myuser"
    password = "mypass"

    file_list = ssh_crawler(
        hostname=hostname,
        port=port,
        username=username,
        password=password
    )

    display_tables(file_list)


def example_interactive():
    """Example: Interactive mode with credential prompts"""
    print("\nExample 4: Interactive Mode (Prompts for credentials)")
    print("-" * 50)

    hostname = "localhost"
    port = 22

    file_list = ssh_crawler(hostname=hostname, port=port)
    display_tables(file_list)


if __name__ == "__main__":
    print("="*80)
    print("SSH CRAWLER - USAGE EXAMPLES")
    print("="*80)
    print("\nChoose an example to run:")
    print("1. Password Authentication")
    print("2. SSH Key Authentication")
    print("3. Custom Port")
    print("4. Interactive Mode (Recommended)")
    print("0. Exit")

    choice = input("\nEnter your choice (0-4): ").strip()

    examples = {
        "1": example_with_password,
        "2": example_with_ssh_key,
        "3": example_with_custom_port,
        "4": example_interactive
    }

    if choice in examples:
        print("\n" + "="*80)
        examples[choice]()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice!")
