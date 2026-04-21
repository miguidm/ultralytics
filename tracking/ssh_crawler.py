#!/usr/bin/env python3
"""
SSH Crawler - Extracts file information from /usr/bin directory on a remote Linux server
"""

import paramiko
from datetime import datetime
from tabulate import tabulate
import getpass


def ssh_crawler(hostname, port=22, username=None, password=None, key_filename=None):
    """
    Connect to SSH server and extract /usr/bin directory contents

    Args:
        hostname (str): SSH server hostname or IP address
        port (int): SSH server port (default: 22)
        username (str): SSH username (prompts if not provided)
        password (str): SSH password (prompts if not provided and no key)
        key_filename (str): Path to SSH private key file (optional)

    Returns:
        list: List of tuples containing (filename, modification_date_string, modification_date_object)
    """
    if username is None:
        username = input("Enter SSH username: ")

    if password is None and key_filename is None:
        password = getpass.getpass("Enter SSH password (or press Enter to use SSH key): ")
        if not password:
            key_filename = input("Enter path to SSH private key (default: ~/.ssh/id_rsa): ").strip()
            if not key_filename:
                key_filename = None

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    file_list = []

    try:
        print(f"\n[*] Connecting to {hostname}:{port}...")

        if key_filename:
            ssh_client.connect(
                hostname=hostname,
                port=port,
                username=username,
                key_filename=key_filename
            )
        else:
            ssh_client.connect(
                hostname=hostname,
                port=port,
                username=username,
                password=password
            )

        print("[+] Connected successfully!")
        print("[*] Extracting /usr/bin directory contents...")

        command = 'find /usr/bin -maxdepth 1 -type f -exec stat -c "%n|%Y" {} \\;'
        stdin, stdout, stderr = ssh_client.exec_command(command)

        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

        if error and not output:
            print(f"[-] Error executing command: {error}")
            return []

        for line in output.strip().split('\n'):
            if line:
                parts = line.split('|')
                if len(parts) == 2:
                    filepath, timestamp = parts
                    filename = filepath.split('/')[-1]
                    mod_date = datetime.fromtimestamp(int(timestamp))
                    mod_date_str = mod_date.strftime('%Y-%m-%d %H:%M:%S')
                    file_list.append((filename, mod_date_str, mod_date))

        print(f"[+] Extracted {len(file_list)} files from /usr/bin")

    except paramiko.AuthenticationException:
        print("[-] Authentication failed. Please check your credentials.")
    except paramiko.SSHException as e:
        print(f"[-] SSH error: {e}")
    except Exception as e:
        print(f"[-] Error: {e}")
    finally:
        ssh_client.close()

    return file_list


def display_tables(file_list):
    """
    Display two tables: sorted by filename and sorted by modification date

    Args:
        file_list (list): List of tuples containing (filename, mod_date_str, mod_date_obj)
    """
    if not file_list:
        print("[-] No files to display.")
        return

    table_data = [(name, date_str) for name, date_str, _ in file_list]

    print("\n" + "="*80)
    print("TABLE 1: Files sorted by NAME (A-Z)")
    print("="*80)
    sorted_by_name = sorted(table_data, key=lambda x: x[0].lower())
    print(tabulate(
        sorted_by_name,
        headers=['Filename', 'Modification Date'],
        tablefmt='grid',
        showindex=range(1, len(sorted_by_name) + 1)
    ))

    print("\n" + "="*80)
    print("TABLE 2: Files sorted by MODIFICATION DATE (Newest to Oldest)")
    print("="*80)
    sorted_by_date = sorted(file_list, key=lambda x: x[2], reverse=True)
    sorted_by_date_display = [(name, date_str) for name, date_str, _ in sorted_by_date]
    print(tabulate(
        sorted_by_date_display,
        headers=['Filename', 'Modification Date'],
        tablefmt='grid',
        showindex=range(1, len(sorted_by_date_display) + 1)
    ))

    print("\n" + "="*80)
    print(f"Total files: {len(file_list)}")
    print("="*80)


def main():
    """
    Main function to run the SSH crawler
    """
    print("="*80)
    print("SSH CRAWLER - /usr/bin Directory Information Extractor")
    print("="*80)

    hostname = input("\nEnter SSH server hostname/IP: ").strip()
    if not hostname:
        print("[-] Hostname is required!")
        return

    port_input = input("Enter SSH port (default: 22): ").strip()
    port = int(port_input) if port_input else 22

    file_list = ssh_crawler(hostname, port)

    if file_list:
        display_tables(file_list)
    else:
        print("\n[-] No files extracted. Please check your connection and permissions.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Operation cancelled by user.")
    except Exception as e:
        print(f"\n[-] Unexpected error: {e}")
