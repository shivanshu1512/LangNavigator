---
title: Test Document
---

# Getting Started

This is the getting started guide for the test framework.

## Installation

Install the package using pip:

```bash
pip install test-framework
```

Make sure you have Python 3.11 or later.

## Quick Start

Here's a basic example:

```python
from test_framework import App

app = App()
app.run()
```

This creates a new app and runs it.

### Configuration Options

You can configure the app with these options:

- `debug`: Enable debug mode
- `port`: Set the port number

## Advanced Usage

For advanced users, you can customize the pipeline:

```python
from test_framework import Pipeline

pipeline = Pipeline(
    steps=[
        preprocess,
        transform,
        postprocess,
    ]
)
result = pipeline.execute(data)
```

# API Reference

## App Class

The `App` class is the main entry point.

### Methods

#### run()

Starts the application server.

```python
app = App(port=8080)
app.run()  # Starts on port 8080
```

#### stop()

Stops the application gracefully.

## Pipeline Class

The `Pipeline` class chains processing steps.

# Troubleshooting

## Common Errors

### ImportError

If you see `ImportError: No module named 'test_framework'`, make sure you installed it:

```bash
pip install test-framework
```

### Connection Refused

Check that the port is not already in use.
