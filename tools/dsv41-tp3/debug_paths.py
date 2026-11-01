"""Read DWARF 2-4 source paths from a CUDA ELF without loading/executing it."""
import struct


def source_paths(data):
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected little-endian ELF64')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, header[6] + i * header[11])
                for i in range(header[12])]
    names = sections[header[13]]
    strings = data[names[4]:names[4] + names[5]]
    lines = None
    for section in sections:
        name = strings[section[0]:].split(b'\0', 1)[0]
        if name == b'.debug_line':
            lines = data[section[4]:section[4] + section[5]]
            break
    if lines is None:
        return []
    paths, offset = [], 0
    while offset < len(lines):
        length, version = struct.unpack_from('<IH', lines, offset)
        if length == 0xffffffff or version not in (2, 3, 4):
            raise ValueError('Unsupported DWARF line format')
        end = offset + 4 + length
        if end > len(lines):
            raise ValueError('Truncated DWARF')
        pos = offset + 10  # unit length + version + header length
        pos += 1 + (version >= 4) + 3  # min instruction, max ops, statement, base, range
        opcode_base = lines[pos]
        pos += opcode_base
        def string():
            nonlocal pos
            last = lines.index(0, pos, end)
            value = lines[pos:last].decode('utf-8')
            pos = last + 1
            return value
        def uleb():
            nonlocal pos
            result, shift = 0, 0
            while pos < end and shift < 70:
                b = lines[pos]
                pos += 1
                result |= (b & 127) << shift
                if not b & 128:
                    return result
                shift += 7
            raise ValueError('Invalid DWARF integer')
        directories = ['']
        while True:
            directory = string()
            if not directory:
                break
            directories.append(directory)
        while True:
            filename = string()
            if not filename:
                break
            directory = uleb()
            uleb()  # modification time
            uleb()  # file size
            prefix = directories[directory]
            paths.append(prefix.rstrip('/') + '/' + filename if prefix else filename)
        offset = end
    return sorted(set(paths))
