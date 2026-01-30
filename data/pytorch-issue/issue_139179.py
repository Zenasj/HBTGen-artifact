def AttrsDescriptorWrapper(
    divisible_by_16=None,
    equal_to_1=None,
    divisibility_value=16,
    equal_to_value=1,
):
    return AttrsDescriptor.from_dict({
        "cls": "AttrsDescriptor",
        "arg_properties": {
            "tt.divisibility": list(divisible_by_16 or []),
            "tt.equal_to": list(equal_to_1 or []),
        },
        "property_values": {
            "tt.divisibility": divisibility_value,
            "tt.equal_to": equal_to_value,
        }
    })