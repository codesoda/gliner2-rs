use gliner2_rs::{
    json::parse_field_spec,
    schema_spec::{FieldDtype, StructureFieldSpec, StructureSpec},
    structures::build_structure_choice_prefix,
};

#[test]
fn parse_field_spec_defaults_to_list() {
    let spec = parse_field_spec("price").expect("parse");
    assert_eq!(spec.name, "price");
    assert_eq!(spec.dtype, FieldDtype::List);
    assert_eq!(spec.choices, Vec::<String>::new());
    assert_eq!(spec.description, None);
}

#[test]
fn parse_field_spec_supports_dtype_and_description() {
    let spec = parse_field_spec("name::str::Event name").expect("parse");
    assert_eq!(spec.name, "name");
    assert_eq!(spec.dtype, FieldDtype::Str);
    assert_eq!(spec.description.as_deref(), Some("Event name"));
}

#[test]
fn parse_field_spec_choices_default_to_str() {
    let spec = parse_field_spec("party_size::[1|2|3|6+]").expect("parse");
    assert_eq!(spec.name, "party_size");
    assert_eq!(spec.dtype, FieldDtype::Str);
    assert_eq!(
        spec.choices,
        vec!["1".to_string(), "2".to_string(), "3".to_string(), "6+".to_string()]
    );
}

#[test]
fn parse_field_spec_choices_and_list_type() {
    let spec = parse_field_spec("dietary::[vegetarian|vegan]::list::Dietary restrictions").expect("parse");
    assert_eq!(spec.name, "dietary");
    assert_eq!(spec.dtype, FieldDtype::List);
    assert_eq!(
        spec.choices,
        vec!["vegetarian".to_string(), "vegan".to_string()]
    );
    assert_eq!(spec.description.as_deref(), Some("Dietary restrictions"));
}

#[test]
fn parse_field_spec_preserves_description_colons() {
    let spec = parse_field_spec("field::str::desc with :: colons").expect("parse");
    assert_eq!(spec.name, "field");
    assert_eq!(spec.dtype, FieldDtype::Str);
    assert_eq!(spec.description.as_deref(), Some("desc with::colons"));
}

#[test]
fn build_structure_choice_prefix_includes_only_choice_fields() {
    let spec = StructureSpec {
        name: "reservation".to_string(),
        fields: vec![
            StructureFieldSpec::new("restaurant").dtype(FieldDtype::Str),
            StructureFieldSpec::new("seating")
                .dtype(FieldDtype::Str)
                .choices(vec!["indoor".to_string(), "outdoor".to_string()]),
            StructureFieldSpec::new("dietary")
                .dtype(FieldDtype::List)
                .choices(vec!["vegetarian".to_string(), "none".to_string()]),
        ],
    };

    let prefix = build_structure_choice_prefix(&spec);
    // Expected: ( reservation: seating ( indoor | outdoor ) , dietary ( vegetarian | none ) )
    assert_eq!(prefix.first().map(String::as_str), Some("("));
    assert_eq!(prefix.get(1).map(String::as_str), Some("reservation:"));
    assert!(prefix.contains(&"seating".to_string()));
    assert!(prefix.contains(&"dietary".to_string()));
    assert!(!prefix.contains(&"restaurant".to_string()));
}

