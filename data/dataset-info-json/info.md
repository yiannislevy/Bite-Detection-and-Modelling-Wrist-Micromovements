# Standardization info

- `mean_std_values.json`: first values, calculated from sequenced samples - unwindowed, gravity not removed, unfiltered.
- `mean_std_values_2.json`: second values, calculated from windowed sequenced samples, gravity removed, filtered.
- `mean_std_values_3.json`: third values, calculated from sequenced samples - unwindowed, gravity removed, filtered.
- `mean_std_values_4.json`: fourth values, calculated from sequenced samples - unwindowed, gravity correctly removed, filtered.
# Signal Start Times

The starting timestamp for each session's data, along with their duration in windows.

- `signal_start_time.json`:  Middle Sample Labeling Approach was used.
- `signal_start_time-MAJORITY-LABELS.json`: Majority Label Approach was used.

# Subject & Index Pairing

This file contains information regarding which session corresponds to which subject. Subject is the key.

- `subject_to_indices.json`: List of sessions paired to the String key-subject_id