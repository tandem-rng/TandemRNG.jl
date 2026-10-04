# Read the cross-check fixtures of tandem-c and tandem-cuda, copied verbatim into
# test/fixtures, so that the files stay byte identical to their sources.

using SHA

const FIXTURES = joinpath(@__DIR__, "fixtures")

fixture_text(name) = read(joinpath(FIXTURES, name), String)
fixture_sha256(name) = bytes2hex(open(sha256, joinpath(FIXTURES, name)))

const C_NUMBER = r"(-?(?:0x[0-9a-fA-F]+|[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?))[uUlLfF]*"

# The brace initializer of the C array `name` as nested vectors of number strings, with the
# integer and float suffixes removed.
function c_initializer(text, name)
    m = match(Regex("\\b$name\\b[^=]*=\\s*\\{"), text)
    items, _ = _c_braces(text, m.offset + length(m.match))
    return items
end

function _c_braces(s, i)
    items = Any[]
    while true
        c = s[i]
        if c == '}'
            return items, i + 1
        elseif c == '{'
            item, i = _c_braces(s, i + 1)
            push!(items, item)
        elseif isspace(c) || c == ','
            i += 1
        else
            m = match(C_NUMBER, s, i)
            m.offset == i || error("unexpected text at $i: $(s[i:min(end, i+20)])")
            push!(items, m[1])
            i += length(m.match)
        end
    end
end

c_scalar(text, name) = match(Regex("\\b$name\\b[^=]*=\\s*([0-9]+)"), text)[1]
