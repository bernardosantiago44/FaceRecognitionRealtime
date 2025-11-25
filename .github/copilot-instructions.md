This is a real-time face identification desktop application built in Python using Tkinter for the UI and face_recognition + dlib for embeddings. 
The system processes live video from one or multiple cameras and maintains a local face database containing:
- Unique person IDs (e.g., persona_0231)
- Face embeddings
- Multiple stored face images (improving quality over time)
- Metadata such as creation date, last_seen, and optional name assigned by the user

Core Functional Requirements
1. Real-time camera feed with high FPS and non-blocking UI updates.
2. Face detection & identification performed in parallel worker threads to avoid blocking the UI.
3. Automatic registration of new, unknown faces (new ID + metadata).
4. Stable tracking using track IDs to avoid wandering bounding boxes.
5. User labeling workflow:
  - User can click a detected face on the live feed to open a modal naming dialog.
  - User can assign or update a person’s name.
  - Names and metadata persist across sessions.
6. People/Profiles tab:
  - Left panel: list of all registered persons (ID, name, thumbnail, last_seen).
  - Right panel: editable profile fields (name, thumbnail, metadata).
7. Metadata schema must remain backward compatible as new fields are added (e.g., names).

Architecture Notes
- Use a PersonRegistry module/class as the single source of truth for all person data:
    - Fetch by ID
    - Update fields (e.g., name)
    - Persist metadata to JSON/disk
    - Emit data-change events for UI refresh
- UI should never write files directly; it must call the registry.
- Camera processing and face identification must run on separate threads. The Tkinter thread should only update the UI.
- When a user assigns a name, update the in-memory registry and then trigger persistence.

Performance Constraints
- Increasing camera resolution must not severely reduce FPS.
- UI interactions (e.g., opening dialogs, switching tabs) must never block recognition threads.
- Thumbnail computation must be lightweight; avoid expensive operations inside the UI thread.

What Copilot Should Generate
Copilot should prioritize:
- Small, atomic changes per file.
- Clear, maintainable functions with minimal side effects.
- UI components that remain responsive under load.
- Thread-safe interaction between recognition threads and UI updates.
- Modular code that makes it easy to expand person profiles with additional metadata fields in the future.

What Copilot Should Avoid
- Blocking the Tkinter mainloop.
- Recomputing embeddings on the UI thread.
- Storing data outside the PersonRegistry.
- Introducing new heavy dependencies without explicit instruction.
