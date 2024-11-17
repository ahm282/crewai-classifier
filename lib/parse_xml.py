import xml.etree.ElementTree as ET
from typing import Dict


def parse_itemgroup(
    element: ET.Element, categories: Dict[str, str], current_path: str = ""
) -> None:
    """
    Recursively parse itemgroups and build complete hierarchical paths down to the deepest level only.

    Args:
        element (ET.Element): Current XML element being processed
        categories (Dict[str, str]): Dictionary to store category codes and their full paths
        current_path (str): Current path being built
    """
    # Get the current group's code and name
    group_code = element.findtext("iigroup", "")
    group_name = element.findtext("iigroupname", "")

    # Build the current path
    path = f"{current_path} > {group_name}" if current_path else group_name

    # Look for nested itemgroups
    itemgroups = element.find("itemgroups")
    if itemgroups is not None:
        # Recur for all subgroups
        for subgroup in itemgroups.findall("itemgroup"):
            parse_itemgroup(subgroup, categories, path)
    else:
        # If no subgroups exist, it's a leaf node, so store the category path
        if group_code:
            categories[group_code] = path


def get_complete_hierarchy(xml_content: str) -> Dict[str, str]:
    """
    Parse XML content and return a dictionary of complete hierarchical paths for the deepest levels only.

    Args:
        xml_content (str): XML content as string

    Returns:
        Dict[str, str]: Dictionary with category codes as keys and their full hierarchical paths as values
    """
    categories = {}

    try:
        # Parse the XML content
        root = ET.fromstring(xml_content)

        # Navigate to the itemgroups section
        # Assuming the structure: tos > tosdata > itemgroups > itemgroup
        tos_data = root.find(".//tosdata")

        if tos_data is not None:
            initial_itemgroups = tos_data.find("itemgroups")
            if initial_itemgroups is not None:
                for itemgroup in initial_itemgroups.findall("itemgroup"):
                    parse_itemgroup(itemgroup, categories)

        return categories

    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}


def get_categories(xml_file_path):
    """
    Reads the XML file, parses it, and returns the deepest hierarchical category paths.

    Args:
        xml_file_path (str): Path to the XML file.

    Returns:
        Dict[str, str]: Dictionary with category codes as keys and their full hierarchical paths as values.
    """
    # Read the XML content
    with open(xml_file_path, "r") as file:
        xml_content = file.read()

    # Get the hierarchy
    return get_complete_hierarchy(xml_content)
