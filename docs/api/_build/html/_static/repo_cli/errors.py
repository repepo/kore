from clyent.errors import ClyentError


class RepoCLIError(ClyentError):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

        if not hasattr(self, "message"):
            self.message = args[0] if args else None


class Unauthorized(RepoCLIError):
    pass


class Conflict(RepoCLIError):
    pass


class InvalidName(RepoCLIError):
    pass


class NotFound(RepoCLIError, IndexError):
    def __init__(self, *args, **kwargs):
        RepoCLIError.__init__(self, *args, **kwargs)
        IndexError.__init__(self, *args, **kwargs)
        self.message = args[0]
        self.msg = args[0]


class UserError(RepoCLIError):
    pass


class ServerError(RepoCLIError):
    pass


class ShowHelp(RepoCLIError):
    pass


class NoMetadataError(RepoCLIError):
    pass


class NoDefaultChannel(RepoCLIError):
    pass


class WrongRepoAuthSetup(RepoCLIError):
    pass


class NoDefaultUrl(RepoCLIError):
    def __init__(self):
        self.msg = (
            "Repository URL is not configured. Please use `conda repo config --set default_site <SITE>`"
            " or refer to the documentation"
        )
        super().__init__(self.msg)


class DestionationPathExists(RepoCLIError):
    def __init__(self, location):
        self.msg = "destination path '{}' already exists.".format(location)
        self.location = location
        super(RepoCLIError, self).__init__(self.msg)


class PillowNotInstalled(RepoCLIError):
    def __init__(self):
        self.msg = (
            "pillow is not installed. Install it with:\n" "    conda install pillow"
        )
        super(RepoCLIError, self).__init__(self.msg)


class ChannelFrozen(RepoCLIError):
    def __init__(self, channel):
        self.msg = "Channel {} is frozen.".format(channel)
        super(RepoCLIError, self).__init__(self.msg)


class BulkActionError(RepoCLIError):
    def __init__(self, code, message):
        self.msg = "Bulk action failed with code: {} and message: {}".format(
            code, message
        )
        super(RepoCLIError, self).__init__(self.msg)


class SystemDiagonseError(RepoCLIError):
    def __init__(self):
        self.msg = (
            "The cleanup job has been started. Check the logs for more information."
        )
        super(RepoCLIError, self).__init__(self.msg)
